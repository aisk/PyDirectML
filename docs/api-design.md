# PyDirectML Python API 设计

## 1. 前提、目标与非目标

**前提：这个库没有外部用户。** 不留弃用别名、不设弃用期、不为向后兼容保留任何东西 —— API 变更直接落地，五个 sample 和 `samples/sdxl/` 一起改。

**目标**：从零设计一套 pythonic 的 API，同时与 DirectML 提供的概念保持一一对应。判据是：`import directml` 之后写出来的代码，光看调用点就能读懂在做什么，不需要同时开着 `DirectMLX.h` 和 `DirectML.h` 对照参数位置。

本文档经历过一轮「只修签名、不动类型结构」的改进，其中不少已经落地（文中以【已落地】标注，部分带 commit）。后来 `samples/sdxl/` 的实践证明：即便签名全部修好，真实使用者仍要手写同一层包装 —— `dml_layers.py` 的 `Model` 类做 index 记账和 binding 收集，`sizes()` / `data_type()` 抹平 getter，`NUMPY_DTYPES` 手抄 dtype 对照表。这些不是领域逻辑，是 API 毛刺在用户代码里的镜像。于是设计升级为现在的形态：**把「人人都会再写一遍」的部分吸收进库，不吸收属于模型领域的部分**（layer 组合、按名字取权重）。

**非目标 —— 明确不做的事**：

- 不引入新抽象。没有 Keras 式的 `Layer` / `Sequential`、没有自动求导、没有算子融合的语法糖、不用字符串替代枚举。
- 不追求补齐算子覆盖。当前只绑定了约 25 个算子，DirectMLX 有上百个；按 sample 的实际需要一个个补，不做批量补齐。目前因此加进来的是 `activation_softmax` 的 `axes` 重载和 `activation_gelu`，都来自 `samples/sdxl/`。
- 不在 Python 层写领域逻辑，但**不禁止 Python 代码**。这条原先写作「不写 Python 逻辑层」，定得太硬，而且现状已经在违反它、方向还是反的：`module.cpp` 的 Binding 构造在 C++ 里 `py::module_::import("numpy")` 再调 `can_cast` —— 用 C++ 写 Python，两边的坏处都占了。改成分层原则：**C++ 拥有资源、执行和数据热路径**（上传/回读、memcpy、persistent resource、`Expression` 的哈希），**Python 拥有签名整形、默认值、校验与报错、dtype 对照、namedtuple、工厂、docstring**。判据是归属和成本，不是语言纯度 —— 库的边界是 `import directml`，不是 `.pyd` 的边界（§4.8）。校验逻辑每次 dispatch 跑一遍也没关系：字典查找是纳秒级，相对一次 GPU dispatch 完全不可见。

## 2. 概念映射

「一一对应」指**概念 1:1**，不是类型 1:1：每个 DML 概念只有一个 Python 对应物，但**纯数据的概念用 Python 原生类型承载**，只有带行为、带资源的概念才是类。

| DML / DirectMLX 概念 | Python 对应物 | 现状与差距 |
| --- | --- | --- |
| `pydml::Device`（D3D12 + `IDMLDevice`） | `dml.Device` | 已有；执行方法移到 `CompiledOperator` 上（§4.7） |
| `dml::Graph` | `dml.Graph` | 现名 `GraphBuilder`，改回 `Graph` 与 C++ 对齐；`compile` 对齐 `Graph::Compile`；构造补 `tensor_policy=` —— 图级 policy 现在进不去（`DirectMLX.h:3437` 走 `builder->GetTensorPolicy()`），中间张量只能是 `Default` |
| `dml::Expression` | `dml.Expression` | 已有；加 `.shape` / `.strides` / `.dtype` / `.desc` 属性并实现可哈希（§4.3） |
| `dml::TensorDesc` | `dml.TensorDesc` | 已有；4 个靠实参类型消歧的重载收敛成单一构造 + 关键字默认值（§4.4） |
| `IDMLCompiledOperator` + persistent resource | `dml.CompiledOperator` | 现名 `Model`，名字误导 —— 它不含权重、不含输入绑定，只是编译好的算子，sample 里的变量名 `op = builder.build(...)` 说明作者自己也不认这个名字。persistent resource 已在它身上【已落地】，`initialize` / `dispatch` 跟着资源走（§4.7） |
| `DML_BUFFER_BINDING` 集合 | **`dict[Expression, np.ndarray]`** | 现为 `Binding` 类 + 按位置对齐的列表；类从公开面退场（§4.1） |
| `pydml::TensorData` | **`np.ndarray`** | dispatch 直接返回形状、dtype 都正确的 numpy 数组；类退场（§4.2） |
| `DML_SIZE_2D` | **`(width, height)` 元组** | `Size2D` 类删除 |
| `DML_TENSOR_DATA_TYPE` | 枚举保留，但**处处兼收 numpy dtype**（`np.float32`、`np.uint32`…） | 每个 sample 都自建 dtype 对照表，这张表该活在库里 —— 按 §1 的分层，它就是 Python 包装层里的一个 dict |
| 11 个枚举 | 带作用域的枚举成员 | 【已落地 40baaad】去掉了全部 `export_values()`。此前 132 个成员平铺到模块顶层，`dml.NONE` / `dml.FORWARD` / `dml.BACKWARD` 三组名字按注册顺序互相覆盖，传错枚举时 `TypeError` 的信息完全对不上因果；`OperatorType` 的 100+ 个成员把 `dml.CONVOLUTION`、`dml.SLICE` 这些通用词全部占掉，`dir(dml)` 有 180 项，删后剩 55 项 |
| `dml::MaxPoolingOutputs` / `GRUOutputs` | **namedtuple** | 可解包；未请求的输出是 `None`，不是空 `Expression`（§4.6） |
| `dml::FusedActivation` | `dml.FusedActivation` + 18 个静态工厂 | 工厂一个都没绑（§4.5） |
| `dml::GRUOutputOptions` | `dml.GRUOutputOptions` | 现名 `OutputOptions`，被削短；它只服务于 GRU，改回全名 |
| DirectMLX 算子自由函数 | 模块级 snake_case 函数 | 已有；签名规则见 §4.5 |

## 3. 核心 API 一览

```python
import numpy as np
import directml as dml

device = dml.Device()                      # use_gpu=True, use_debug_layer=False
graph = dml.Graph(device)                  # optional tensor_policy=

# input auto-assigns its index; owned=True is OWNED_BY_DML; dtype takes numpy dtypes
x = graph.input([1, 1, 28, 28])                          # float32 is the default
w = graph.input([8, 1, 5, 5],  np.float32, owned=True)
b = graph.input([1, 8, 1, 1],  np.float32, owned=True)
s = graph.input([1, 8, 28, 28], strides=[8, 1, 0, 0])    # broadcast view, one line

conv = dml.convolution(x, w, b, strides=[1, 1],
                       start_padding=[2, 2], end_padding=[2, 2])
probs = dml.activation_softmax(conv, axes=[1])

op = graph.compile([probs])                # outputs fixed here, not repeated at dispatch

op.initialize({w: np.load("w.npy"), b: np.zeros([1, 8, 1, 1], np.float32)})
result, = op({x: image})                   # __call__ == dispatch, returns np.ndarray list
```

迭代场景（sdxl 的采样循环）就是 `initialize` 一次、循环里只 `op({...})`。图里没有 owned 输入时 `initialize` 自动完成，所以 matmul 这类一次性计算一行都不用写它 —— 这恰好与 DML 的语义对齐：初始化本来就只为 persistent 权重存在。

## 4. 设计决策

### 4.1 binding 是 `dict[Expression, ndarray]`

整个设计里杠杆最大的一条。它要解决的问题是当前 API 最容易出错、也最不显眼的地方 —— 输入绑定顺序是一份隐式契约：

```python
x = dml.input_tensor(builder, 0, desc_x)     # ← 序号写在这
w = dml.input_tensor(builder, 1, desc_w)

device.compute(op, [dml.Binding(x, ...), dml.Binding(w, ...)], [out])
#                   ↑ bindings[i] 必须对应上面的 input_tensor(builder, i, ...)
```

`Device::DispatchOperator` 纯按下标配对，既不校验数量也不校验对应关系。序号写错或者 bindings 排错，**不报错**，只是权重错位、输出是垃圾。4 个 sample 各自抄了同一个 helper 来绕开它（`mnist.py:43-49` 用 `len(input_bindings)` 当序号）；superres.py 手写 `0..8` 外加一张手工维护的对照表。

dict 方案不是给这份契约加护栏，而是让它不存在，一次解决四个问题：

- **index 隐式契约彻底消失**。dict 按 Expression 配对，`Graph` 内部记录每个 input 的 index，错位在结构上不可能发生；缺输入、多输入在 dispatch 时抛 `ValueError` 并点名是哪个 expression。
- **CPU 拷贝不复存在**。`Binding` 会给每个权重留一份 CPU 拷贝（UNet 半精度 5.1 GiB，曾为此加了 `release_data()`）；dict 方案下数据在 `initialize` / `dispatch` 调用的那一刻直接从调用者的数组上传，库不保留任何拷贝。`release_data` 连同「释放后再 dispatch 会炸」的契约一起删除。
- **dtype 校验规则原样保留，挪到上传点执行**，报错能带上具体是哪个输入。规则是「**同 kind，或者 NumPy 认为 safe**」【已落地】—— 注意 NumPy 自己的 `same_kind` 达不到这个粒度，它放行 int32 → float32，正好是要拦住的那一个：

  | from → to | 同 kind | safe | 结果 |
  | --- | --- | --- | --- |
  | float64 → float32 | 是 | 否 | 放行（`np.zeros` 的默认 dtype，必须放行） |
  | float32 → float16 | 是 | 否 | 放行（半精度权重靠这条加载） |
  | uint8 → float32 | 否 | 是 | 放行（保值） |
  | int32 → float32 | 否 | 否 | **拒绝**，要显式 `astype` —— 安静算错的头号入口 |
  | float32 → int32 | 否 | 否 | 拒绝 |

  字节数校验同样保留【已落地】：数组字节数不得超过 `TotalTensorSizeInBytes`，不足则补齐 —— 原先 `device.cpp` 按张量大小 `memcpy`，只有一句 Release 下为空的 `assert` 挡着。
- **owned 与非 owned 的区分落在 API 结构上**。`initialize` 只收 owned 输入、`dispatch` 只收非 owned 输入，dict 的 key 集合就是校验规则本身。

实现要点：

- `Expression` 按 `NodeOutput*` 指针实现 `__hash__` / `__eq__`，pybind 里两行。机会成本记一笔：`ELEMENT_WISE_LOGICAL_EQUALS` 就在枚举里，将来若要 torch 风格的逐元素 `a == b`，`__eq__` 这个名字已经被占 —— 届时开 `dml.equals()` 函数即可，dict 绑定的收益远大于这个保留。这是有意为之，不是没想到。
- **`{Expression: (index, desc, owned)}` 映射在 `compile` 时快照进 `CompiledOperator`**，让 op 自包含 —— 用户的自然写法是 compile 完就丢掉 graph，`initialize` / `dispatch` 不得隐式要求 graph 还活着。key 只做身份比对，永不解引用，所以 graph 销毁后指针值继续当哈希键用是安全的。传入一个不是 input 的 Expression（图的中间节点）抛 `ValueError`。
- **逐张量转换 + 上传**。float64 → float32、非连续数组在上传点仍会产生一次瞬时转换拷贝，这没问题；有问题的是先把整个 dict 转换完再上传 —— 对 UNet 5.1 GiB 的权重，峰值内存会把 `release_data` 解决过的问题原样请回来。

### 4.2 dispatch 直接返回 `np.ndarray`

现在每个调用点都要写 `np.array(output_data[0], np.float32).reshape(...)`，而形状、dtype 库全都知道。C++ 侧从 readback heap 拷进按 desc 构造好的 numpy 数组返回即可。

`TensorData` 只出现在 `Device::Compute` 的返回路径上，没有任何 API 收它作输入，它存在的唯一理由是 C++ 需要一块 buffer —— 这不构成公开类型的理由。它按 dtype 决定 `itemSize` 和 buffer format 的修复【已落地】保留在内部：此前 `itemSize` 写死 `sizeof(float)`，对一个 UINT8 输出张量，Python 侧会按 `4N` 字节去读一块只有 `N` 字节的堆内存，是真正的越界读，不只是「按 float 重新解释」。

### 4.3 `Expression` 的属性与可哈希

加 `.shape`（tuple）、`.strides`、`.dtype`（numpy dtype）、`.desc`（完整 `TensorDesc`）。`dml_layers.py` 开头的 `sizes()` 和 `data_type()` 两个 helper 就是这条的直接证据。`get_output_desc()` 这种 getter 换成属性。`__repr__` 输出 `<dml.Expression float16 [1, 64, 512, 512]>`，调试图结构时省一半功夫。

默认构造已删除【已落地 40baaad】—— `Expression::Impl()` 的 `m_nodeOutput` 是 nullptr，`dml.Expression().get_output_desc()` 曾直接段错误。

### 4.4 `graph.input(...)` 与 `TensorDesc` 的收敛

`input_tensor(builder, index, desc)` 变成 `Graph` 的方法 —— 概念仍是 `dml::InputTensor`，但归属关系摆到语法上，index 由 graph 自动分配。签名：

```python
graph.input(sizes=None, dtype=np.float32, *, owned=False, strides=None, desc=None)
```

常用路径不碰 `TensorDesc` 和 `TensorFlags`；要精确控制（`total_tensor_size_in_bytes`、`guaranteed_base_offset_alignment`）就传完整 `desc=`，此时其余参数一律非法（`TypeError`，`sizes` 也不例外）—— desc 里已经写了一遍的东西再收一遍，就得回答「哪个赢」，而这正是 §4.1 刚消灭掉的那类隐式契约。`TensorDesc` 本身收敛成一个构造函数加关键字默认值。

`TensorPolicy` 的两个静态属性已修好【已落地 40baaad】（原先 getter 少了类对象参数，Python 侧完全拿不到实例）：

```python
>>> dml.TensorDesc(np.float32, [1, 3, 8, 8], tensor_policy=dml.TensorPolicy.interleaved_channel).strides
[192, 1, 24, 3]                      # NHWC
```

但它只对单个 `TensorDesc` 生效，图内部生成的中间张量走 `Graph` 的构造参数 —— 所以 `dml.Graph(device, tensor_policy=...)` 是 `InterleavedChannel` 真正可用的前提。

### 4.5 算子签名

**位置参数只留张量，其余全部 kw-only**（pybind11 的 `py::kw_only()`）。这条规则来自真实 sample 里的反例：

```python
device = dml.Device(True, True)                       # 哪个 True 是啥？
dml.batch_normalization(conv1, mean, variance, scale, bias,
                        1, 0.000009999999747378752, ...)          # 1 是 spatial
dml.mean_variance_normalization(conv4, scale, bias, [0, 2, 3],
                                1, 1, 0.000009999999747378752, ...)  # 两个 1 各是啥？
```

而所有 sample 对张量之后的参数本来就在用关键字（`strides=[2,2], start_padding=[1,1]`），规则和现有代码零冲突。

**默认值与参数顺序**。`epsilon` 在 5 个 sample 里全是同一个魔数 `0.000009999999747378752`（ONNX 的 `1e-5` 的 float32 表示），默认 `1e-5f`；`normalize_variance` / `normalize_mean` 默认 `True`；GRU 的 `direction` 默认 `FORWARD`、`output_options` 默认 `BOTH`。必需参数不再排在可选参数之后（`mean_variance_normalization` 和 `gru` 都有三个连续必需参数跟在可选参数后面，逼出了上面那行最难读的调用）。注意 `mean_variance_normalization` 的 `normalize_mean` 和 `average_pooling` 的 `dilations` 在 C++ 里是用 `DML_TARGET_VERSION` 条件编译插在参数表中间的，绑定位置与 C++ 逐字一致，不动。

**命名对齐 C++**：`up_sample_2d` → `upsample_2d`（C++ 是 `Upsample2D` 一个词）、`join(input=)` → `inputs=`、`OutputOptions` → `GRUOutputOptions`、`Model` → `CompiledOperator`。`activation_soft_max` → `activation_softmax` 已落地（同时补了 `axes` —— 旧绑定走 `DML_ACTIVATION_SOFTMAX`，对 4-D 张量直接 `CreateOperator` 失败，attention 用不了）。另外补齐所有缺失的 `py::arg` —— `compute` / `build` / `Size2D.__init__` 的参数现在只有 `arg0` / `arg1`，而 compute 是这个库唯一的执行入口。

**`reinterpret(x, sizes, strides=None, dtype=None)`**，`dtype=None` 表示不变。sdxl 里十几处 `dml.reinterpret(x, data_type(x), ...)` 的 `data_type(x)` 全是噪音，因为绝大多数 reinterpret 不改 dtype。

**`FusedActivation` 绑 18 个静态工厂**。sample 里出现 20 多次的 `dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU)` 变成 `dml.FusedActivation.relu()`；带默认参数的 `leaky_relu(alpha=0.01)`、`hard_sigmoid(alpha=0.2, beta=0.5)` 等照 DirectMLX 逐个搬。纯 1:1 补全，没有设计自由度，顺带把 `FusedActivation(dml.OperatorType.CONVOLUTION)` 这种非法组合排除在外。按 §1 的分层落在 Python 包装层，一个工厂三行。

**`average_pooling` 补上漏绑的 `output_sizes`**（`DirectMLX.h:2348`，`convolution` 绑了对应参数，纯遗漏）。

**docstring 统一补 Args / Returns / Raises**，写在 Python 包装层（§4.8），不是 C++ 字符串字面量。优先 `Graph.compile`、`graph.input`、`CompiledOperator.initialize` / `dispatch`，以及参数含义不能望文生义的算子（`local_response_normalization` 的 `alpha` / `beta` / `bias` 现在只能去查 `DirectML.h`）。

### 4.6 多输出算子返回 namedtuple，未请求的输出是 `None`

`max_pooling(...)` 返回 `MaxPoolingOutputs(values, indices)`；`output_indices=False` 时 `indices is None`，而不是一个碰一下就段错误的空 `Expression`（DirectMLX 此时返回 `{ outputExpr, Expression() }`，`GetOutputDesc()` 对它解引用空指针 —— 这是最后一个能让解释器崩溃的入口，在这个表示下不存在）。支持 `values, indices = dml.max_pooling(..., output_indices=True)`。GRU 同理。

namedtuple 用 `typing.NamedTuple` 定义在 Python 包装层，`_core` 返回普通元组 —— pybind11 没有原生 namedtuple 支持，C++ 侧要 import collections 手工拼，这是 §1 分层原则最具体的受益者。

### 4.7 执行模型：`initialize` / `dispatch` 挂在 `CompiledOperator` 上

历史：`Device::Compute` 原本每次调用都重跑一遍初始化，且 persistent resource 挂在 `Device` 上 —— 单个缓冲区，初始化第二个算子会覆盖第一个的内容，「初始化一次、之后只分派」在同一个 device 上建两个模型时根本不成立。这两点已修【已落地】：persistent resource 移到编译好的算子上（连同它的 `ResourceAllocator` 引用，否则 Python 先销毁 `Device` 再销毁算子时段错误），`Initialize` / `Dispatch` 拆开。实测收益（RX 6800，`samples/sdxl` 解码，重复调用稳态）：

| 分辨率 | 每次 initialize + dispatch | 只 dispatch | |
| --- | --- | --- | --- |
| 512x512 | 0.33 s | 0.16 s | 2.1x |
| 1024x1024 | 1.09 s | 0.70 s | 1.6x |

拆开看，大头是不再重跑初始化，不是权重常驻：同一个二进制上只切换 `OWNED_BY_DML` 标志做对照，512 是 184 ms → 162 ms，1024 是 758 ms → 731 ms，只占总收益一两成。权重那一项跟权重字节数走，初始化那一项跟图的规模走。（别拿单次冷调用算这笔账 —— 首次分派要付缓冲区分配和 meta-command 建立，1024 下 9 s 以上，只付一次。）

本设计在此之上只做归属调整：方法从 `Device` 移到 `CompiledOperator` —— 资源在谁身上，方法就在谁身上。`op.initialize(weights)`、`op.dispatch(inputs)`、`op(...)` 作为 dispatch 的别名；`temporary_size` / `persistent_size` / `descriptor_count` 属性保留。`compute`（初始化 + 分派合二为一）删除 —— 有了「无 owned 输入免 initialize」的规则，它没有存在价值了。

随之而来的契约保留：`OWNED_BY_DML` 张量的数据在 `initialize` 时被读走，之后换数据要重新 `initialize`，docstring 里写明。

另一条契约收紧：**图里有 owned 输入而没 initialize 就 dispatch，抛错并点名缺哪些 owned expression**，不沿用现在首次 dispatch 隐式初始化的行为。「无 owned 输入免 initialize」（§3）是唯一的自动路径 —— 那种图的 initialize 是空操作，自动做掉没有歧义；有权重而忘了给，是错误不是默认值。

### 4.8 打包成 package，发类型存根

`directml/` 包内放 `_core.pyd`，`__init__.py` 就是 §1 分层原则里 Python 侧的落点：re-export `_core` 的类和枚举，再放签名整形的包装函数、namedtuple 定义、`FusedActivation` 工厂、dtype 对照表和校验报错。包装层是普通 Python 代码，签名和 docstring 自文档，mypy 直接读；`py.typed` 加 pybind11-stubgen 只需要对付 `_core` 里剩下的类。IDE 补全和 mypy 从零变一。代价是同一个函数的行为分两处看（包装层的整形 + `_core` 的执行），用「包装层不做领域逻辑、只做整形」的纪律控制住。

配套的实现细节（对 `_core` 仍然成立）：`py::class_` 注册整体移到 `module.def` 之前，类与类之间按依赖排序 —— pybind11 在 `def` 的那一刻渲染签名字符串，未注册的类型退回 C++ 原始名，现在 docstring 里漏着 `pydml::CompiledModel`、`dml::Expression` 这种 Python 里不合法的名字，存根也会跟着坏。

`__version__` 保留：正经打包之后它几乎是白送的（从包元数据读），不再经 CXXFLAGS 走一遍。

## 5. 效果对照

以 mnist.py 的前半段为例。改造前：

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
...
op = builder.build(dml.ExecutionFlags.NONE, [softmax])
output_data = device.compute(op, input_bindings, [softmax])
output_tensor = np.array(output_data[0], np.float32)
```

改造后（helper 整个消失，绑定错位从「垃圾输出」变成结构上不可能）：

```python
device = dml.Device(use_gpu=True, use_debug_layer=True)
graph = dml.Graph(device)

x = graph.input([1, 1, 28, 28])
w = graph.input([8, 1, 5, 5], owned=True)
b = graph.input([1, 8, 1, 1], owned=True)
conv = dml.convolution(x, w, b, strides=[1, 1], start_padding=[2, 2], end_padding=[2, 2])
...
op = graph.compile([softmax])
op.initialize({w: np.load(path / "Parameter5.npy"), b: np.zeros([1, 8, 1, 1], np.float32)})
probs, = op({x: rescaled_image})
```

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

sdxl 的 `dml_layers.Model` 里属于**库的缺陷**的部分全部蒸发：`_add_input` 的编号记账、`NUMPY_DTYPES`、`placeholder` 的 shape/dtype 记录与校验、`run` 里的 reshape、`release_data` 循环。剩下的只有**属于模型的**东西 —— 按名字取权重、决定谁是 constant 谁是 placeholder —— 大约 20 行，而且这 20 行本来就该由应用层写：

```python
class Model:
    def __init__(self, device, dtype=np.float32):
        self.graph = dml.Graph(device)
        self.weights = {}                        # Expression -> ndarray
        self.dtype = dtype

    def constant(self, array, shape=None):
        array = np.ascontiguousarray(array, self.dtype)
        expr = self.graph.input(shape or array.shape, self.dtype, owned=True)
        self.weights[expr] = array
        return expr

    def compile(self, outputs):
        self.op = self.graph.compile(outputs)
        self.op.initialize(self.weights)
        self.weights.clear()                     # the library keeps no copy; this was the only one
        return self
```

`broadcast` / `to_tokens` / `split_heads` 这些 stride 技巧函数保留在 sample 层 —— 它们是领域知识，不是 API 毛刺，吸进库里就违反「不引入新抽象」。

## 6. 落地顺序

已落地的部分在上文就地标注，汇总：去掉全部 `export_values()`、删除不可达 API（`Dimensions`、`Binding` 的空 `buffer_protocol` 标注、三个无消费者的构造函数、`Expression` 默认构造）、修好 `TensorPolicy` 静态属性（以上 40baaad）；`TensorData` 按 dtype 决定 `itemSize` 与 buffer format（修掉越界读）、`Binding` 的 dtype 与字节数校验；`initialize` / `dispatch` 拆分与 persistent resource 归属；`activation_softmax` 改名并补 `axes`；附录里的杂项修复。

剩余工作按依赖排。两处顺序调整：拆包提前 —— 包装层是 namedtuple、工厂、docstring 的落点，得先存在；sample 改写不再押后到最后一步 —— 第 4-6 步都是破坏性变更，而 sample 是这个仓库唯一的回归手段，**sample 随每一步就地迁移**（§1 的前提「变更和 sample 一起改」本来就这么要求，原顺序和它矛盾，会留一段什么都跑不起来的窗口）：

1. 类注册顺序整理 + 补齐缺失的 `py::arg`（§4.8 的配套，`_core` 的 docstring 和存根都受它影响）。
2. 拆包：`directml/` + `_core.pyd` + 包装层骨架（§4.8）。存根可以最后发，但分层得先立起来。
3. `Expression` 的属性和 `__hash__` / `__eq__`（§4.1 / §4.3）—— dict 绑定依赖它。
4. `CompiledOperator`（改名 + `initialize(dict)` / `dispatch(dict)` / `__call__`，dispatch 返回 numpy 数组）；删除 `Device` 上的 `compute` / `initialize` / `dispatch`、`Binding` / `TensorData` 的公开注册和 `release_data`（§4.1 / §4.2 / §4.7）。**配一个 pytest**：cast 表五行、缺输入/多输入报错、字节数补齐、漏 initialize 的报错 —— 仓库目前没有测试，samples 兼职回归，而这些规则恰恰最容易被后续改动无声破坏，§4.1 的表本身就是现成的测试用例。
5. `graph.input(...)`、`Graph` 改名并收 `tensor_policy=`、`TensorDesc` 收敛构造（§4.4）。
6. 算子签名：`py::kw_only()`、默认值与参数顺序、改名、`average_pooling` 的 `output_sizes`、`reinterpret` 新签名、`Size2D` → 元组（§4.5）；`FusedActivation` 工厂和 namedtuple 输出落在包装层（§4.5 / §4.6）。
7. docstring 的 Args / Returns / Raises（包装层）+ 存根（§4.8）。
8. `dml_layers.Model` 缩到 §5 的形态（其余 sample 改写已随第 4-6 步就地完成）。

## 7. 已拍板的争议项

原先这里是三个未定项，现在都定了：

1. **`Graph`**，不是 `GraphBuilder`。与 `dml::Graph` 名字对齐是本文档一贯的判据，`compile()` 方法已把 builder 语义说清楚，不在这一个名字上破例。
2. **枚举不收字符串**（`mode="nearest"` 这种 torch 风格）。这本来就是 §1「不用字符串替代枚举」那条非目标，不是独立的未定项；枚举可 grep、可补全，字符串是又一套要维护的对照表。
3. **`graph.input` 用 `owned=True` 布尔**，不留 `flags=`。`TensorFlags` 实际只有 `NONE` 和 `OWNED_BY_DML` 两个值；为假想的第三个 flag 预留接口，正是本文档在别处反对的做法。真加了第三个 flag 再补也不迟。

## 附录：顺带发现的非 API 问题

不属于本文档范围，但既然读到了就记一笔。

- **【已修复 40baaad】** `device.cpp:168`：`if (!desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)` —— 运算符优先级问题，实际算的是 `(!desc.flags) & FLAG`，即 `flags == 0` 时得 `1 & 1 == 1`，`flags != 0` 时得 `0`。应为 `if (!(desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML))`。当前 `flags` 只有 `NONE`(0) 和 `OWNED_BY_DML`(1) 两个取值，结果碰巧一致，但只要加入第三个标志位就会出错。
- **【已删除 40baaad】** 原 `module.cpp:209`：`.def("build", [](dml::Graph& self, ...) { self; return new ...; })` 里孤立的 `self;` 语句是无用的（大概是为了消 unused 警告，但 `self` 明明被 `CompiledModel` 用了）。
- **【已删除 40baaad】** `setup.py:52` 把 `VERSION_INFO` 塞进 `CXXFLAGS`，但 `src/` 里一次都没引用（pybind11 项目模板的残留）。删得对；不过 `__version__` 本身要有 —— §4.8 正经打包之后它几乎是白送的，从包元数据读，不走 CXXFLAGS。
- **【已修复】** `device.cpp` 的 `EnsureUploadHeapSize` / `EnsureReadBackHeapSize` / `EnsureDefaultBufferSize` / `EnsureCpuBufferSize` 都用 `RoundUpToPow2` 做几何增长。小缓冲区没问题，但过了 GiB 量级就有两个后果：一是白白多要一倍内存，二是 2.1 GiB 的请求会取整成 **4 GiB 的单个 resource**，分配它会直接把设备干掉（`DXGI_ERROR_DEVICE_REMOVED`），而不是返回 `E_OUTOFMEMORY`。阈值卡得很准 —— `samples/sdxl` 的 OpenCLIP ViT-bigG 权重 2.00 GiB 能过、2.07 GiB 挂。现在超过 256 MiB 改成按固定步长增长（`util.h` 的 `GrowBufferSize`）。
- **【已修复】** `util.h` 的 `ThrowIfFailed(HRESULT)` 抛的是不带消息的 `std::exception`，到 Python 侧显示为 `RuntimeError: Unknown exception` —— 既看不出是哪个 HRESULT，也分不清是设备调用失败还是分配失败。上面那个 bug 就是因为这个多花了一轮才定位到。现在把 HRESULT 打进消息里。
- **【已确认是硬限制，非本仓库问题】** 单个 D3D12 buffer 到 **4 GiB** 就到头了 —— 再大不是返回 `E_OUTOFMEMORY`，而是直接 `DXGI_ERROR_DEVICE_REMOVED`。阈值实测很干净：一张带 3.75 GiB 权重的图能初始化，4.00 GiB 的不能，且和显存余量无关（空出 14 GiB 再测一样）。这条约束会往上传导：DirectML 把一个模型的 `OWNED_BY_DML` 权重折进**单个** persistent resource，所以**任何超过 4 GiB 权重的模型都建不成一张图**。`samples/sdxl` 的 UNet 半精度 4.78 GiB，只能在 mid block 处劈成两张图（2.31 + 2.47 GiB）跑。上面那条缓冲区增长的修复只是不再主动撞这堵墙，墙本身还在。
- **【已加，将随 §4.1 一起删除】** `Binding.release_data()`：`Binding` 会给每个权重留一份 CPU 拷贝，但 `OWNED_BY_DML` 的数据在初始化后就归 DirectML 了，这份拷贝纯属浪费。dict 绑定落地后库不再保留任何拷贝，这个方法和它服务的问题一起消失。`device.cpp` 里「缓冲区长度和张量对不上」的检查（原先是 Release 下为空的 `assert`）保留。
