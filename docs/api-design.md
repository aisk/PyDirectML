# PyDirectML Python API 改进设计

## 1. 前提、目标与非目标

**前提：这个库当前没有外部用户。** 所以不留弃用别名、不设弃用期、不为向后兼容保留任何东西 —— 改名就直接改名，同步改掉 5 个 sample 即可。既有代码里凡是只为兼容而存在的，一并删掉。

**目标**：让 `import directml` 之后写出来的代码，光看调用点就能读懂在做什么 —— 不需要同时开着 `DirectMLX.h` 和 `DirectML.h` 对照参数位置。

**非目标 —— 明确不做的事**：

- 不做大规模重构。C++ 侧的 `Device` / `CompiledModel` / `Binding` / `TensorData` 结构不动。（**这条后来推翻了**，见 §3.13：persistent resource 必须从 `Device` 移到 `CompiledModel`，否则「初始化一次、之后只分派」做不成。）
- 不引入新抽象。**概念映射保持 1:1**：`dml::Graph`、`dml::Expression`、`dml::TensorDesc`、`IDMLCompiledOperator`、`DML_BUFFER_BINDING` 在 Python 侧各自仍然只有一个对应类型。不做 Keras 式的 `Layer` / `Sequential`，不做自动求导，不做算子融合的语法糖。
- 不追求补齐算子覆盖。当前只绑定了约 25 个算子，DirectMLX 有上百个；那是另一件事。
- 不加 Python 包装层。全部改动落在 `module.cpp` / `model.h` / `device.cpp`，扩展模块保持顶层单文件。理由见 §4。

一句话概括方向：**同样的概念，更好的签名。**

## 2. 现状的概念映射

| DirectML / DirectMLX | 当前 Python | 评价 |
| --- | --- | --- |
| `pydml::Device`（D3D12 + `IDMLDevice`） | `Device` | 名字对，构造参数是裸布尔 |
| `dml::Graph` | `GraphBuilder` | 名字比 C++ 还清楚，保留 |
| `dml::Expression` | `Expression` | 对（凭空造空对象的构造已删除，见 §3.8） |
| `dml::TensorDesc` | `TensorDesc` | 对，但 4 个重载靠实参类型消歧 |
| `dml::TensorDimensions` | ~~`Dimensions`~~ | 类型分发层面不可达，已删除，见 §3.7 |
| `dml::TensorPolicy` | `TensorPolicy` | 静态属性已修好，但图级 policy 仍进不去，见 §3.7 |
| `IDMLCompiledOperator` | `Model` | 名字有歧义，且完全不透明 |
| `pydml::Binding` | `Binding` | 对，但丢了「第几个输入」这个信息 |
| `pydml::TensorData` | `TensorData` | 对，但 dtype 被写死成 float32 |
| `dml::FusedActivation` | `FusedActivation` | 只绑了通用构造，丢了 18 个静态工厂 |
| `dml::GRUOutputOptions` | `OutputOptions` | 名字被削短，和 C++ 对不上 |

映射本身是健康的，问题几乎全在**签名层**。这是好消息：绝大部分改进不改变行为。

## 3. 问题清单

### 3.1 `export_values()` 造成模块级名字冲突 【已落地 40baaad】

`module.cpp` 里 11 个枚举中有 10 个调用了 `.export_values()`（唯一没调用的是 `OutputOptions`），把成员名平铺到模块命名空间。它的实现是 `m_parent.attr(name) = value`，后注册的无声覆盖先注册的。三组名字直接撞车：

| 模块级名字 | 来源（按注册顺序） | `dml.X` 实际拿到的 |
| --- | --- | --- |
| `dml.NONE` | `TensorFlags` → `MatrixTransform` → `ExecutionFlags` | `ExecutionFlags.NONE` |
| `dml.FORWARD` | `RecurrentNetworkDirection` → `ConvolutionDirection` | `ConvolutionDirection.FORWARD` |
| `dml.BACKWARD` | `RecurrentNetworkDirection` → `ConvolutionDirection` | `ConvolutionDirection.BACKWARD` |

后果不是算错。pybind11 2.10.3 的 `enum_` 不给枚举注册任何隐式转换，所以 `dml.gru(..., direction=dml.FORWARD)` 传进去一个卷积方向枚举会抛 `TypeError`。真正的问题是这个 `TypeError` 的信息完全对不上因果：调用点写的名字看起来是对的，报错说的却是「incompatible argument types」，然后列出一堆和 `ConvolutionDirection` 无关的重载。

另外 `OperatorType` 有 100+ 个成员全部平铺，`dml.INVALID`、`dml.CONVOLUTION`、`dml.GEMM`、`dml.SLICE` 这些通用词全部占用了模块顶层（`dir(dml)` 有 180 项）。

注意 132 个枚举成员只占掉不到 132 个顶层名字 —— 平铺时同名互相覆盖。删掉后 `dir(dml)` 剩 55 项，全是类型名、函数名和 dunder。

**改进**：全部去掉 `export_values()`，只保留 `dml.TensorFlags.NONE` 这种带作用域的写法。所有 sample 已经是这么写的（`dml.TensorDataType.FLOAT32`、`dml.PaddingMode.REFLECTION`），没有一处依赖平铺名，所以这是零成本改动。

### 3.2 `Binding` 把任何 dtype 强制转成 float32 【已落地】

```cpp
// module.cpp:172
.def(py::init([](dml::Expression& expression,
                 py::array_t<float, py::array::c_style | py::array::forcecast> data) {
```

`forcecast` 的意思是：传 `int32` 数组进来，NumPy 会做数值转换（`3` → `3.0f`），再按 float32 的位模式上传到 GPU。而 tensor 的 `TensorDataType` 可能声明的是 `INT32`。这不是报错，是安静地算错。

输出侧更严重 —— 写死了 float32：

```cpp
// model.h:41-44
TensorData(dml::TensorDesc* desc) :
    itemSize(sizeof(float)),
    format(py::format_descriptor<float>::format()),
```

`Device::Compute` 的输出全部走这个构造函数（`device.cpp:398`）。而 `buffer` 只 `resize` 到 `desc->totalTensorSizeInBytes`（`model.h:72`），暴露给 buffer 协议的 shape 却是 `desc->sizes`、`itemSize` 是 4。对一个 UINT8 输出张量，Python 侧会按 `4N` 字节去读一块只有 `N` 字节的堆内存 —— 这是**越界读**，不只是「按 float 重新解释」。

**改进**：

- `TensorData(TensorDesc*)` 按 `desc->dataType` 查表得到 `itemSize` 和 buffer 协议 format 字符串。这张表只有 11 项，一次写完。
- `Binding` 接受 `py::array`（不指定元素类型），构造时用 `expression.GetOutputDesc().dataType` 做校验。

**校验要放行安全转换**。`mnist.py:46` 的 `np.zeros(tensor.get_output_desc().sizes)` 产生的是 float64，forcecast 成 float32 是**正确**的；一刀切拒绝会让现在正确的代码报错。用 `np.can_cast(arr.dtype, target, 'same_kind')` 放行、其余抛 `TypeError` 并写明期望的 dtype，是合适的粒度。跨类别的（int32 → float32）必须显式写 `arr.astype(...)`。

这是**唯一的内存安全问题**，优先级最高。

**落地时的一处修正**：上面写的 `np.can_cast(arr.dtype, target, 'same_kind')` 达不到本节自己要的粒度。NumPy 的 `same_kind` 除了「同类别内的窄化」，还放行**向上跨类别**的转换，`np.can_cast(int32, float32, 'same_kind')` 返回 `True` —— 正好是本节点名要拦住的那一个。实际落地的判据是「**同 kind，或者 NumPy 认为 safe**」：

| from → to | 同 kind | safe | `same_kind` | 实际 |
| --- | --- | --- | --- | --- |
| float64 → float32 | 是 | 否 | 是 | 放行（`np.zeros` 的默认 dtype，必须放行） |
| float32 → float16 | 是 | 否 | 是 | 放行（半精度权重靠这条加载） |
| uint8 → float32 | 否 | 是 | 是 | 放行（保值） |
| int32 → float32 | 否 | 否 | **是** | **拒绝**，要显式 `astype` |
| float32 → int32 | 否 | 否 | 否 | 拒绝 |

另外补了一条本节没提到的越界：`Binding` 现在校验数组字节数不超过 `TotalTensorSizeInBytes`，并把缓冲区补齐到该长度。原先 `device.cpp` 是按 `TotalTensorSizeInBytes` 从数组里 `memcpy` 的，只有一句 `assert` 挡着 —— 而 `assert` 在 Release 下是空的。

### 3.3 布尔和标量的裸位置参数

来自真实 sample 的调用：

```python
device = dml.Device(True, True)                       # mnist.py:52

dml.batch_normalization(conv1, mean, variance, scale, bias,
                        1, 0.000009999999747378752,   # mobilenet.py:66
                        dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

dml.mean_variance_normalization(conv4, scale, bias, [0, 2, 3],
                                1, 1, 0.000009999999747378752,  # candy.py:59
                                dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

dml.value_scale_2d(input, 1.0, scaler1_bias)          # candy.py:46
dml.slice(instance_norm11, [0,0,2,2], [1,64,196,196], [1,1,1,1])   # candy.py:102
dml.local_response_normalization(x, cross_channel, local_size, alpha, beta, bias)
```

`Device(True, True)` 的两个 `True` 是 `use_gpu` 和 `use_debug_layer`；`batch_normalization` 里的 `1` 是 `spatial`；`mean_variance_normalization` 里连续两个 `1` 分别是 `normalize_variance` 和 `normalize_mean`。这些都是布尔，被写成 `1`，因为 pybind 的 `bool` 参数接受 int。

**改进**（两条并行）：

1. **对布尔和无量纲标量参数强制关键字**。pybind11 支持 `py::kw_only()`：把它插在 `py::arg("bias")` 之后，后面所有参数就必须用关键字传。`dml.Device(True, True)` 会变成写不出来的东西，必须写 `dml.Device(use_gpu=True, use_debug_layer=True)`。
2. **给显然的默认值补上默认值**。`epsilon` 在 5 个 sample 里全是同一个字面量 `0.000009999999747378752`（也就是 ONNX 的 `1e-5` 的 float32 表示）。让它默认 `1e-5f`，调用点直接消失一个魔数。

哪些参数应该 kw-only，建议规则：**位置参数只留张量**。`convolution(input, filter, bias)` 三个张量走位置，`strides` 往后全部关键字 —— 这恰好也是所有 sample 已经在写的风格（`dml.convolution(input, w, b, strides=[2,2], start_padding=[1,1])`）。规则和现有代码零冲突。

### 3.4 必需参数排在可选参数后面

Python 允许，但结果是调用者被迫二选一：要么全用关键字，要么把前面的默认值也全部手写一遍。

```cpp
// module.cpp:411-418  mean_variance_normalization
py::arg("input"),
py::arg("scale") = dml::NullOpt,
py::arg("bias") = dml::NullOpt,
py::arg("axes") = std::vector<uint32_t>{},
py::arg("normalize_variance"),        // 必需，却排在可选之后
py::arg("normalize_mean"),            // 同上
py::arg("epsilon"),                   // 同上
py::arg("fused_activation") = dml::FusedActivation::None());
```

三个连续的必需参数跟在三个可选参数后面。`gru` 也一样：`bias` / `hidden_init` / `sequence_lengths` 有默认值，紧跟其后的 `activation_descs` / `direction` / `output_options` 没有（`module.cpp:566-575`）。

candy.py 的写法就是被这个逼出来的 —— 它必须把 `[0,2,3]`、`1`、`1`、`1e-5` 全部按位置排好，这正是 §3.3 里最难读的那一行。

**改进**：`normalize_variance` / `normalize_mean` 默认 `True`，`epsilon` 默认 `1e-5`；GRU 的 `direction` 默认 `FORWARD`、`output_options` 默认 `Both`、`activation_descs` 保持必需但提到 `recurrence` 之后。配合 §3.3 的 kw-only，调用点变成：

```python
dml.mean_variance_normalization(conv4, scale, bias,
                                axes=[0, 2, 3],
                                fused_activation=dml.FusedActivation.relu())
```

注意 `mean_variance_normalization` 的 `normalize_mean`（`DirectMLX.h:3284`）和 `average_pooling` 的 `dilations`（`DirectMLX.h:2345`）在 C++ 里本来就是用 `DML_TARGET_VERSION` 条件编译插在参数表中间的，当前绑定的位置与 C++ 逐字一致，这部分不需要动。

### 3.5 参数名不准确、缺失，或与 C++ 不一致

| 位置 | 现在 | 应为 | 说明 |
| --- | --- | --- | --- |
| `module.cpp:316` | `input_tensor(scope=...)` | `graph=` | 传的是 `GraphBuilder`，不是什么 scope（`DirectMLX.h:1238` 形参名就是 `graph`） |
| `module.cpp:549` | `join(input=[...])` | `inputs=` | 收的是列表，C++ 也叫 `inputs` |
| `module.cpp:536` | `reinterpret(new_size=)` | `new_sizes=` | 收的是列表，且 `TensorDesc` 那边叫 `sizes` |
| `module.cpp:351` | `up_sample_2d` | `upsample_2d` | C++ 是 `Upsample2D`，一个词 |
| ~~`module.cpp:539`~~ | ~~`activation_soft_max`~~ | `activation_softmax` | C++ 是 `ActivationSoftmax`，一个词。**已落地**，同时补了 `axes` 参数 —— 旧绑定只调 `DML_ACTIVATION_SOFTMAX`，它对 4-D 张量直接 `CreateOperator` 失败，attention 用不了；`DML_ACTIVATION_SOFTMAX1` 收 `axes`。见 `samples/sdxl/` |
| `module.cpp:42` | `OutputOptions` | `GRUOutputOptions` | C++ 类型名就是这个，且它只服务于 GRU |
| `module.cpp:203` | `Model` | `CompiledOperator` | 它是 `IDMLCompiledOperator`，不是「模型」 |

`Model` 这个名字尤其误导：`builder.build(...)` 返回的东西叫 `Model`，但它不含权重、不含输入绑定，只是编译好的算子。sample 里的变量名是 `op = builder.build(...)`，说明作者自己也不认这个名字。

全部直接改名，不留旧名，同步改 5 个 sample。

**另有 3 处根本没写 `py::arg`**，参数在 Python 侧只有 `arg0` / `arg1`：

| 位置 | 签名现状 |
| --- | --- |
| `module.cpp:182` | `Device.compute(self, arg0, arg1, arg2)` |
| `module.cpp:199` | `GraphBuilder.build(self, arg0, arg1)` |
| `module.cpp:293` | `Size2D.__init__(self, arg0, arg1)` |

`compute` 是这个库唯一的执行入口，它的三个参数连名字都没有。原先还有 `TensorData.__init__` 和 `MaxPoolingOutputs.__init__`，已随 §3.7 一起删除。

### 3.6 类注册顺序导致签名里泄漏 C++ 原始类型名

pybind11 在 `def` 的那一刻就把签名渲染成字符串，此时还没注册的类型只能退回原始 C++ 名。`module.cpp` 里 `Binding`(171)、`Device`(178)、`GraphBuilder`(194) 全都排在 `CompiledModel`(203)、`TensorData`(255)、`Expression`(266) 之前，于是：

```
compute(self, arg0: pydml::CompiledModel, arg1: List[directml.Binding],
        arg2: List[dml::Expression]) -> List[pydml::TensorData]
build(self, arg0: directml.ExecutionFlags, arg1: List[dml::Expression]) -> pydml::CompiledModel
Binding.__init__(self, expr: dml::Expression, data: numpy.ndarray[numpy.float32])
```

`dml::Expression` 和 `pydml::CompiledModel` 在 Python 里根本不是合法名字。既然不发类型存根（§4），这些 docstring 签名就是用户能拿到的**唯一**类型信息源，坏得更不能接受。

**改进**：把所有 `py::class_` 注册整体移到 `module.def` 之前，类与类之间也按依赖排序（`Expression`、`TensorData`、`CompiledModel` 在 `Binding`、`Device`、`GraphBuilder` 之前）。纯移动，零行为变化。

### 3.7 不可达和多余的 API 面 【已落地 40baaad，`Model` 改名除外】

原标题「死掉和不可达的 API 面」不准确，说的也不是同一件事。下面第一组是真的用不了，第二组能正常工作、只是没有消费者。两组都删了，但理由不同。

**一、本身不可用**

- **`Dimensions`**（原 `module.cpp:215`）没有任何构造函数或方法，`dml.Dimensions()` 抛 `No constructor defined!`。关键不在缺构造函数 —— 补上也没用：没有 absl 时 `TensorDimensions` 就是 `std::vector<uint32_t>`（`DirectMLX.h:317, 361`），而 `pybind11/stl.h:215` 的 `type_caster<std::vector<Type, Alloc>>` 是偏特化，恒胜通用的 `type_caster_base`，所以类型转换器永远不会查询这个注册类型。**这是类型分发层面的不可达。**
- **`Binding` 的 `py::buffer_protocol()`**（原 `module.cpp:181`）声明了却没有 `def_buffer`。pybind 照样把 `tp_as_buffer` 装上，但 `get_buffer` 是空的，于是 `memoryview(binding)` 抛 `BufferError: pybind11_getbuffer(): Internal error`。

  这条有两个修法，不是只能删：`pydml::Binding` 确实持有一个完整的 `TensorData data`（`model.h:109`，有 buffer / shape / strides），补 `def_buffer` 也能让标注成立。选了删标注 —— `Binding` 的语义是「输入绑定描述」，它持有的那份 buffer 是上传前的拷贝，暴露出去只会让人误以为能从 `Binding` 读回计算结果。

**二、能用，但没有消费者**

`MaxPoolingOutputs.__init__`（原 322）、`GRUOutputs.__init__`（原 329）、`TensorData.__init__`（原 268）。这三个都能正常构造，`TensorData` 的 buffer 协议也确实读得回来 —— 它们不是 bug。问题是 grep 整个 `src/` 确认：`TensorData` 只出现在 `Device::Compute` 的返回路径（`device.cpp:135` / `398`），另两个只由 pooling 和 GRU 算子返回，**没有任何 API 收它们作输入**。构造出来无处可传，所以是多余的 API 面。

删掉的代价是：如果以后想让 `compute` 接受预分配的输出缓冲，`TensorData` 的构造函数正好是现成入口，届时重新加回。

`Expression.__init__`（原 282）不属于这一类 —— 它构造出来的对象一用就崩，见 §3.8。

**三、需要修，不是删**

**`TensorPolicy`**（现 `module.cpp:205-207`）两个静态属性的 getter 写成了零参 lambda，而 `def_property_readonly_static` 会把类对象当第一个参数传进去：

```
>>> dml.TensorPolicy.default
TypeError: (): incompatible function arguments
>>> dml.TensorPolicy.interleaved_channel
TypeError: (): incompatible function arguments
```

加上没绑构造函数，**Python 侧完全拿不到一个 `TensorPolicy` 实例**，`TensorDesc` 的 `tensor_policy=` 参数永远只能是默认值。已改成 `[](py::object) { ... }`。验证：

```python
>>> dml.TensorDesc(f32, [1, 3, 8, 8], tensor_policy=dml.TensorPolicy.interleaved_channel).strides
[192, 1, 24, 3]                      # NHWC
>>> dml.TensorDesc(f32, [1, 3, 8, 8]).strides
None                                 # 打包 NCHW
```

但**修好后它只对单个 `TensorDesc` 生效**：图级 policy 走 `dml::Graph` 的构造参数（`DirectMLX.h:3437` 用的是 `builder->GetTensorPolicy()`），而 `GraphBuilder` 只绑了 `device`，policy 进不去 —— 图内部生成的中间张量一律是 `Default`。`InterleavedChannel` 要真正用起来，还得给 `GraphBuilder` 补一个 `tensor_policy` 参数。这一条本文档原先没有覆盖，也尚未落地。

**四、保留**

**`Model` 空类**（现 `module.cpp:203`）—— 不删，改名 `CompiledOperator`（§3.5）并加个 `__repr__`。尚未落地。

### 3.8 空 `Expression` 能拿到手，用了就崩 【默认构造已删除 40baaad；`max_pooling` 入口未处理】

两个入口。一个是 `max_pooling`：

```python
pooling66 = dml.max_pooling(relu32, strides=[2, 2], window_sizes=[2, 2])
convolution110 = dml.convolution(pooling66.values, ...)     # mnist.py:74, 79
```

`output_indices` 默认 `False`，此时 DirectMLX 返回的是 `{ outputExpr, Expression() }`（`DirectMLX.h:2489`）—— 第二个是默认构造的空 `Expression`。而 `MaxPoolingOutputs.indices` 是 `def_readwrite`，Python 侧照样读得出来。

另一个是 `.def(py::init<>())`（原 `module.cpp:282`），让 Python 能凭空造一个空 `Expression`。

`Expression::Impl()` 返回的 `m_nodeOutput` 是 nullptr，`GetOutputDesc()` 直接解引用它（`DirectMLX.h:852`）：

```
$ python -c "import directml as dml; dml.Expression().get_output_desc()"
Segmentation fault
```

这是唯一能让解释器崩溃的 API 面。

**改进**：

- 保持 `MaxPoolingOutputs` 的返回类型不变（1:1 映射 `dml::MaxPoolingOutputs`，也不破坏 `.values` 的现有写法），但在上面记录 `output_indices`，`.indices` 无效时抛 `ValueError` 并说明原因。另外给它加 `__iter__`，支持 `values, indices = dml.max_pooling(..., output_indices=True)`。
- ~~去掉 `Expression` 的默认构造绑定。~~ **已落地** —— Python 侧没有任何合法理由需要造一个不属于任何图的表达式，`dml.Expression()` 现在抛 `No constructor defined!`。
- `max_pooling` 那个入口仍在：`output_indices=False` 时 `.indices` 依然读得出一个空 `Expression`，只是现在没法再从 Python 凭空造第二个。

### 3.9 输入绑定顺序是隐式契约

这是当前 API 最容易出错、也最不显眼的地方。

```python
x = dml.input_tensor(builder, 0, desc_x)     # ← 序号写在这
w = dml.input_tensor(builder, 1, desc_w)
b = dml.input_tensor(builder, 2, desc_b)

device.compute(op, [dml.Binding(x, ...), dml.Binding(w, ...), dml.Binding(b, ...)], [out])
#                   ↑ bindings[i] 必须对应上面的 input_tensor(builder, i, ...)
```

`Device::DispatchOperator` 就是纯按下标配对（`device.cpp:156-177`），既不校验数量也不校验对应关系。这个约束在 API 上没有任何体现 —— `Binding` 持有 expression，但不知道自己的 index。序号写错或者 bindings 排错，**不报错**，只是权重错位、输出是垃圾。

4 个 sample 各自抄了同一个 helper 来绕开它（`mnist.py:43-49`：用 `len(input_bindings)` 当序号）；superres.py 则是手写 `0..8` 外加一张手工维护的对照表（`superres.py:41-81`）。

**改进**：把 index 的记账放进 `GraphBuilder` —— 在 `pydml` 侧给 `dml::Graph` 包一层，持有下一个可用 index 和一张 `NodeOutput* → index` 的映射：

1. `input_tensor(graph, tensor_desc, input_index=None)` —— 省略时自动取下一个，显式给也仍然合法。
2. `Binding` 构造时从这张表查出自己的 index 并记下来。
3. `compute` 校验这批 binding 的 index 是不是恰好构成 `0..n-1`，有重复 / 缺失 / 数量不符就抛 `ValueError`。

第 1 条让 4 个 sample 的 helper 直接消失；第 3 条让 superres.py 那种显式写法出错时当场报错，而不是产生垃圾输出。这不是新概念 —— `GraphBuilder` 映射的仍然是 `dml::Graph`，只是多记了一个 `dml::InputTensor` 本来就需要的 `inputIndex`。

### 3.10 `FusedActivation` 丢了 18 个静态工厂

sample 里出现了 20 多次的写法：

```python
dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU)
```

DirectMLX 的 `FusedActivation` 本身带 18 个静态工厂 —— 无参的 `None()` / `Relu()` / `Sigmoid()` / `Tanh()` / `Identity()` / `Softsign()` / `Gelu()`，带正确默认参数的 `Elu(alpha=1.0)` / `LeakyRelu(alpha=0.01)` / `HardSigmoid(alpha=0.2, beta=0.5)` / `ScaledElu(...)` / `ScaledTanh(alpha=1.0, beta=0.5)` / `Softplus(steepness=1.0)` / `ThresholdedRelu(alpha=1.0)` / `Shrink(bias=0.0, threshold=0.5)` / `Celu(alpha=1.0)`，以及必传参数的 `Linear(alpha, beta)` / `ParametricSoftplus(alpha, beta)`。绑定一个都没暴露。

**改进**：把这些静态工厂逐个绑成 `staticmethod` —— `dml.FusedActivation.relu()`、`dml.FusedActivation.leaky_relu(0.01)`、`dml.FusedActivation.none()`。这是纯粹的 1:1 映射补全，没有任何设计自由度，是整份文档里性价比最高的一条。

顺带的好处：现在的通用构造允许 `FusedActivation(dml.OperatorType.CONVOLUTION)` 这种明显非法的组合，工厂方法把合法集合限死了。

### 3.11 docstring 没有契约

`Device.compute` 的 docstring 是 `"Calculate the output of the operator from the input data."`。没说返回值是什么（`list[TensorData]`）、顺序如何（对应 `outputs` 参数）、dtype 是什么、生命周期如何。所有 sample 都要写 `np.array(output_data[0], np.float32)` 才敢用。

算子函数的 docstring 大多是一句从 DirectML 文档抄来的算子描述，没有一个说明参数含义 —— `local_response_normalization` 的 `alpha` / `beta` / `bias` 分别是什么，只能去查 `DirectML.h`。

既然不发类型存根（§4），docstring 就是用户能拿到的唯一说明。

**改进**：统一补 Args / Returns / Raises。优先补 `compute`、`build`、`input_tensor`、`Binding`，以及参数含义不能望文生义的那几个算子。

### 3.12 `average_pooling` 漏绑了 `output_sizes`

`dml::AveragePooling` 的最后一个参数是 `TensorDimensions outputSizes = {}`（`DirectMLX.h:2348`），绑定里没有（`module.cpp:488-505`）。`convolution` 就绑了对应的 `output_sizes`。纯遗漏，补上即可。

### 3.13 `compute` 每次调用都重跑一遍初始化 【已落地，且推翻了 §1 的一条非目标】

`Device::Compute` 把初始化和分派绑在一起，注释自己也说了这是权宜之计：

```cpp
// device.cpp:141
// Ideally initialize only needs to happen once while dispatch occurs every time a new input is bound.
// But for now, we'll do both in one go for each compute call for simplicity.
InitializeOperator(op, inputs);
return DispatchOperator(op, inputs, outputs);
```

初始化是**每个算子一次**的事：它读走带 `DML_TENSOR_FLAG_OWNED_BY_DML` 的输入，让 DirectML 按自己想要的布局折进 persistent resource，之后每次分派都从那里读。重跑它只是把同一批权重再传一遍。对现有 sample 无所谓 —— 它们都只 `compute` 一次；对任何要迭代的东西（扩散采样一次 20~50 步）就是主要成本。

拦路的不是那两行，是 **`m_persistentResource` 挂在 `Device` 上**。它是单个缓冲区，初始化第二个算子会覆盖第一个算子的内容，所以「初始化一次、之后只分派」在同一个 device 上建两个模型时根本不成立。这正是 §1 里「`Device` / `CompiledModel` 结构不动」那条非目标要拦的改动 —— **这条非目标推翻了**，理由是不动结构就做不成这件事，而这件事挡着整个 `samples/sdxl/` 的第二阶段。

落地的样子：

- persistent resource 移到 `CompiledModel`，一并存 `initialized` 标志。`CompiledModel` 还要持有创建它的 `ResourceAllocator` —— gpgmm 释放 allocation 时要回到 allocator，而 Python 完全可能先销毁 `Device` 再销毁 `Model`（模块字典按插入顺序清理，sample 里 `device` 基本都先于 `op` 定义）。不持有就是退出时段错误。
- `Device` 增加 `Initialize` / `Dispatch` 两个公开方法。`Compute` 保留，语义变成「没初始化过就先初始化，然后分派」，所以现有 sample 一行不改，白拿一次提速。
- 代价是一条新契约：`OWNED_BY_DML` 张量的数据在**首次** `compute` 时被读走，之后改 `Binding` 不再生效，要重新 `initialize`。docstring 里写了。

实测（RX 6800，`samples/sdxl` 解码，重复调用的稳态耗时，与改动前的二进制对比）：

| 分辨率 | 改动前（每次 initialize + dispatch） | 改动后（只 dispatch） | |
| --- | --- | --- | --- |
| 512x512 | 0.33 s | 0.16 s | 2.1x |
| 1024x1024 | 1.09 s | 0.70 s | 1.6x |

拆开看，这里面**大头是不再重跑初始化，不是权重常驻**。在同一个二进制上只切换 `OWNED_BY_DML` 标志做对照：512 是 184 ms → 162 ms，1024 是 758 ms → 731 ms，也就是总收益里只占一两成。原因是标志位省的是「重传权重」，量级跟权重字节数走（VAE 只有 320 MiB）；而重跑初始化的开销跟图的规模走。UNet fp32 是 10.3 GiB，权重那一项要乘 32 倍，两项都会放大。

注意别拿单次冷调用来算这笔账。冷调用里首次分派要付掉缓冲区分配和 meta-command 建立，1024 下能到 9 s 以上，两个版本都一样 —— 改动后它只是挪进了 `compile()` 且只付一次。

## 4. 决定不做的事

- **不发 `.pyi` / `py.typed`，因此不拆包。** PEP 561 的内联存根机制只对 package 生效，顶层单文件扩展模块没有对应方案（唯一出路是另发一个 `-stubs` 分发包）。等 Python 社区给出更好的答案再说。连带的结果：扩展模块保持顶层 `directml.pyd`，`setup.py` / `CMakeLists.txt` 不动，也不会有 `__init__.py`。
- **不加 `__version__`，不加 `max_feature_level()`。** 没有必要。
- **不加 `InputCollector` 这类 Python 侧的辅助类。** 它只是绕过 §3.9 的问题，真正的修法是在 C++ 里做 index 记账和校验。
- **不留任何弃用别名。** 见 §1。

## 5. 落地顺序

全部落在 `module.cpp` / `model.h` / `device.cpp`，没有 Python 层，也不需要弃用期。按依赖和风险排：

**第一步：零行为变化的整理**

1. 类注册顺序（§3.6）—— 先做，后面所有 docstring 都受它影响。
2. ~~删掉 `Dimensions`、`Binding` 的 `buffer_protocol` 标注、三个用不到的构造函数；修好 `TensorPolicy` 的静态属性（§3.7）。~~ **已落地 40baaad**
3. 修正参数名，补齐 `compute` / `build` 的 `py::arg`（§3.5），同步改 5 个 sample。
4. ~~去掉全部 `export_values()`（§3.1）。~~ **已落地 40baaad** —— 11 个枚举、132 个成员，运行时确认无一泄漏到模块顶层。
5. 绑定 `FusedActivation` 的 18 个静态工厂（§3.10），同步改 sample。
6. 补上 `average_pooling` 的 `output_sizes`（§3.12）。
7. 补 docstring 的 Args / Returns（§3.11）。

**第二步：正确性**

8. `TensorData` 按 dtype 决定 `itemSize` 和 buffer format（§3.2）—— 修掉越界读。
9. `Binding` 按 `TensorDesc.data_type` 校验 dtype，去掉 `forcecast`（§3.2）。放行 `same_kind` 转换，否则 5 个 sample 的 `np.zeros(...)` 会一起报错。
10. `MaxPoolingOutputs.indices` 的有效性校验 + `__iter__`（§3.8）。默认构造已随 40baaad 去掉。

**第三步：签名**

11. 插入 `py::kw_only()`，位置参数只留张量（§3.3）。
12. 补默认值并重排必需参数（§3.4）。
13. `GraphBuilder` 记 index，`input_tensor` 的 `input_index` 变可选，`compute` 校验（§3.9）。

## 6. 改造前后对照

以 mnist.py 的前半段为例。

改造前：

```python
device = dml.Device(True, True)
builder = dml.GraphBuilder(device)

input_bindings = []
def append_input_tensor(builder, input_bindings, input_tensor, file_name):
    tensor = dml.input_tensor(builder, len(input_bindings), input_tensor)
    if file_name == "":
        input_bindings.append(dml.Binding(tensor, np.zeros(tensor.get_output_desc().sizes)))
    else:
        input_bindings.append(dml.Binding(tensor, np.load(tensor_data_path + "/" + file_name)))
    return tensor

data_type = dml.TensorDataType.FLOAT32
flags = dml.TensorFlags.OWNED_BY_DML
input = dml.input_tensor(builder, 0, dml.TensorDesc(data_type, [1, 1, 28, 28]))
input_bindings.append(dml.Binding(input, rescaled_image))

w = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [8, 1, 5, 5]), "Parameter5.npy")
b = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1, 8, 1, 1]), "")
conv = dml.convolution(input, w, b, strides=[1, 1], start_padding=[2, 2], end_padding=[2, 2])
```

改造后（`input_index` 省略即自动递增，helper 整个消失）：

```python
device = dml.Device(use_gpu=True, use_debug_layer=True)
builder = dml.GraphBuilder(device)

f32 = dml.TensorDataType.FLOAT32
owned = dml.TensorFlags.OWNED_BY_DML

input = dml.input_tensor(builder, dml.TensorDesc(f32, [1, 1, 28, 28]))
w     = dml.input_tensor(builder, dml.TensorDesc(f32, owned, [8, 1, 5, 5]))
b     = dml.input_tensor(builder, dml.TensorDesc(f32, owned, [1, 8, 1, 1]))
conv  = dml.convolution(input, w, b, strides=[1, 1], start_padding=[2, 2], end_padding=[2, 2])

bindings = [
    dml.Binding(input, rescaled_image),
    dml.Binding(w, np.load(path / "Parameter5.npy")),
    dml.Binding(b, np.zeros([1, 8, 1, 1], np.float32)),
]
```

`bindings` 排错或者少一项，`compute` 会抛 `ValueError`，而不是算出垃圾。

candy.py 里最难读的那一行：

```python
# 前
instance_norm5 = dml.mean_variance_normalization(
    conv4, instance_norm5_scale, instance_norm5_bias, [0,2,3], 1, 1,
    0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# 后
instance_norm5 = dml.mean_variance_normalization(
    conv4, instance_norm5_scale, instance_norm5_bias,
    axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())
```

概念一个没变：还是 `Device`、`Graph`、`TensorDesc`、`InputTensor`、`Binding`、`Convolution`。少掉的只是 index 记账和裸布尔。

## 附录：顺带发现的非 API 问题

不属于本文档范围，但既然读到了就记一笔。

- **【已修复 40baaad】** `device.cpp:168`：`if (!desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)` —— 运算符优先级问题，实际算的是 `(!desc.flags) & FLAG`，即 `flags == 0` 时得 `1 & 1 == 1`，`flags != 0` 时得 `0`。应为 `if (!(desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML))`。当前 `flags` 只有 `NONE`(0) 和 `OWNED_BY_DML`(1) 两个取值，结果碰巧一致，但只要加入第三个标志位就会出错。
- **【已删除 40baaad】** 原 `module.cpp:209`：`.def("build", [](dml::Graph& self, ...) { self; return new ...; })` 里孤立的 `self;` 语句是无用的（大概是为了消 unused 警告，但 `self` 明明被 `CompiledModel` 用了）。
- **【已删除 40baaad】** `setup.py:52` 把 `VERSION_INFO` 塞进 `CXXFLAGS`，但 `src/` 里一次都没引用（pybind11 项目模板的残留）。既然不做 `__version__`，这行也可以删掉。
