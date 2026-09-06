# PyDirectML Python API 设计

## 1. 前提、目标与非目标

**前提：这个库没有外部用户。** 不留弃用别名、不设弃用期、不为向后兼容保留任何东西。API 变更直接落地，五个经典 sample 和 `samples/sdxl/` 随每一步一起改，它们是这个仓库除 `tests/` 之外唯一的回归手段。

**目标**：一套 pythonic 的 API，与 DirectML 提供的概念一一对应。判据是：`import directml` 之后写出来的代码，光看调用点就能读懂在做什么，不需要同时开着 `DirectMLX.h` 和 `DirectML.h` 对照参数位置。

**吸收的边界**：把「每个使用者都会再写一遍」的部分吸收进库，不吸收属于模型领域的部分。前者是 index 记账、dtype 对照表、shape/dtype 的 getter、广播视图、按名字绑定；后者是 layer 组合、按名字取权重、attention 的布局约定（`samples/sdxl/dml_layers.py` 里的 `to_tokens` / `split_heads` 留在 sample 层就是这条线）。

**分层原则**：**C++ 拥有资源、执行和数据热路径**（上传/回读、memcpy、persistent resource、`Expression` 的哈希），**Python 拥有签名整形、默认值、校验与报错、dtype 对照、namedtuple、工厂、docstring**。判据是归属和成本，不是语言纯度；库的边界是 `import directml`，不是 `.pyd` 的边界（§4.8）。校验逻辑每次 dispatch 跑一遍没有关系：字典查找是纳秒级，相对一次 GPU dispatch 完全不可见。

**非目标**：

- 不引入新抽象。没有 Keras 式的 `Layer` / `Sequential`、没有自动求导、没有算子融合的语法糖、不用字符串替代枚举。
- 不追求补齐算子覆盖。当前绑定了约 25 个算子，DirectMLX 有上百个；按 sample 的实际需要一个个补。
- 不做隐式的事：不隐式广播、不隐式初始化、不隐式把 GPU 数据拷回 CPU。

## 2. 概念映射

「一一对应」指**概念 1:1**，不是类型 1:1：每个 DML 概念只有一个 Python 对应物，但**纯数据的概念用 Python 原生类型承载**，只有带行为、带资源的概念才是类。

| DML / DirectMLX 概念 | Python 对应物 | 说明 |
| --- | --- | --- |
| `pydml::Device`（D3D12 + `IDMLDevice`） | `dml.Device` | 只负责资源与执行机制；执行方法在 `CompiledOperator` 上（§4.7） |
| `dml::Graph` | `dml.Graph` | `input` / `constant` / `compile`；构造收 `tensor_policy=`，这是图内部中间张量走 `InterleavedChannel` 的唯一入口 |
| `dml::Expression` | `dml.Expression` | `.shape` / `.strides` / `.dtype` / `.size` / `.desc`；按节点身份可哈希（§4.3） |
| `dml::InputTensor` | `graph.input(...)` | index 由 graph 分配（§4.4） |
| `dml::InputTensor` + 已知的权重数据 | `graph.constant(array)` | 声明 owned 输入并记下数据，`compile` 上传（§4.11） |
| `dml::TensorDesc` | `dml.TensorDesc` | 单一构造 + 关键字默认值，收 numpy dtype（§4.4） |
| `IDMLCompiledOperator` + persistent resource | `dml.CompiledOperator` | `initialize` / `dispatch` / `__call__`（§4.7） |
| `DML_BUFFER_BINDING` 集合 | **`dict[Expression | str, ndarray | Buffer]`** | 按 Expression 或 `name=` 配对，不按位置（§4.1、§4.13） |
| `ID3D12Resource`（DEFAULT 堆上的一块张量） | `dml.Buffer` | GPU 常驻张量：`dispatch(readback=False)` 的输出、绑定字典里的值、`constant()` 的实参（§4.10） |
| 回读的输出 | **`np.ndarray`** | dispatch 直接返回形状、dtype 都正确的数组（§4.2） |
| `DML_SIZE_2D` | **`(width, height)` 元组** | |
| `DML_TENSOR_DATA_TYPE` | `dml.TensorDataType`，且**处处兼收 numpy dtype** | 对照表是包装层里的一个 dict |
| 11 个枚举 | 带作用域的枚举成员 | 不 `export_values()`：`dml.NONE` / `dml.FORWARD` 这类平铺名字会按注册顺序互相覆盖 |
| `dml::MaxPoolingOutputs` / `GRUOutputs` | **namedtuple** | 未请求的输出是 `None`（§4.6） |
| `dml::FusedActivation` | `dml.FusedActivation` + 18 个静态工厂 | `FusedActivation.relu()`（§4.5） |
| DirectMLX 算子自由函数 | 模块级 snake_case 函数 | 签名规则见 §4.5 |

## 3. 核心 API 一览

```python
import numpy as np
import directml as dml

device = dml.Device()                      # use_gpu=True, use_debug_layer=False
graph = dml.Graph(device)                  # optional tensor_policy=

x = graph.input([1, 1, 28, 28], name="image")            # float32 is the default
w = graph.constant(np.load("w.npy"))                     # owned input, data recorded now
b = graph.constant(np.load("b.npy"), sizes=[1, 8, 1, 1])
s = graph.input([1, 8, 28, 28], strides=[8, 1, 0, 0])    # a broadcast view fed per dispatch

conv = dml.convolution(x, w, strides=[1, 1],
                       start_padding=[2, 2], end_padding=[2, 2])
conv = conv + dml.broadcast(b, conv.shape)               # explicit, zero-stride, no copy
probs = dml.activation_softmax(conv, axes=[1])

op = graph.compile([probs])                # outputs fixed here; constants uploaded, op initialized

result, = op({"image": image})             # __call__ == dispatch, returns np.ndarray list
gpu, = op({x: image}, readback=False)      # ...or a dml.Buffer that stays on the GPU
next_op({y: gpu})                          # bound in place by the next graph
```

`owned=True` 的输入服务于数据在构图时还没有的情形：`op.initialize({w: array})` 一次，之后循环里只 `op({...})`。`compile` 时若每个 owned 输入都是 constant（包括一个都没有），初始化就地完成。这与 DML 的语义对齐：初始化本来就只为 persistent 权重存在。

## 4. 设计决策

### 4.1 binding 是 `dict[Expression, ndarray]`

整个设计里杠杆最大的一条。上游按位置配对：`input_tensor(builder, i, desc)` 里手写序号，`compute(op, [Binding, ...])` 里按下标对齐，既不校验数量也不校验对应关系。序号写错或列表排错**不报错**，只是权重错位、输出是垃圾。每个 sample 都抄了同一个 helper 来绕开它。

dict 不是给这份契约加护栏，而是让它不存在：

- **index 隐式契约消失**。dict 按 Expression 配对，`Graph` 内部记录每个 input 的 index，错位在结构上不可能发生。缺输入、多输入、绑错阶段、绑两次都在 dispatch 时抛 `ValueError` 并点名是哪个输入（§4.13 的名字也会带上）。
- **CPU 拷贝不存在**。数据在 `initialize` / `dispatch` 调用的那一刻直接从调用者的数组上传，库不保留任何拷贝。UNet 半精度 5.1 GiB 的权重不需要第二份。
- **dtype 校验在上传点执行**，规则是「**同 kind，或者 NumPy 认为 safe**」。NumPy 自己的 `same_kind` 达不到这个粒度，它放行 int32 → float32，正好是要拦住的那一个：

  | from → to | 同 kind | safe | 结果 |
  | --- | --- | --- | --- |
  | float64 → float32 | 是 | 否 | 放行（`np.zeros` 的默认 dtype，必须放行） |
  | float32 → float16 | 是 | 否 | 放行（半精度权重靠这条加载） |
  | uint8 → float32 | 否 | 是 | 放行（保值） |
  | int32 → float32 | 否 | 否 | **拒绝**，要显式 `astype`：安静算错的头号入口 |
  | float32 → int32 | 否 | 否 | 拒绝 |

  字节数同样校验：元素数必须正好填满张量（strided 视图按底层 buffer 的字节数算）；packed 数组因 DirectML 4 字节取整而短几个字节的，上传时补零。
- **owned 与非 owned 的区分落在 API 结构上**。`initialize` 只收 owned 输入、`dispatch` 只收非 owned 输入，dict 的 key 集合就是校验规则本身。

实现要点：

- `Expression` 按 `NodeOutput*` 指针实现 `__hash__` / `__eq__`。机会成本记一笔：将来若要 torch 风格的逐元素 `a == b`，`__eq__` 这个名字已经被占，届时开 `dml.equals()` 函数即可。这是有意为之。
- **`{Expression: (index, desc, owned)}` 在 `compile` 时快照进 `CompiledOperator`**，让 op 自包含：用户的自然写法是 compile 完就丢掉 graph，`initialize` / `dispatch` 不要求 graph 还活着。key 只做身份比对，永不解引用，所以 graph 销毁后指针值继续当哈希键用是安全的。传入一个不是 input 的 Expression（图的中间节点）抛 `ValueError`。
- **逐张量转换 + 上传**。C++ 侧的上传循环拉一个 Python 生成器，每拉一次才转换一个数组，任何时刻只有一份转换拷贝活着。先把整个 dict 转换完再上传，对模型级别的权重会把峰值内存翻倍。

### 4.2 dispatch 直接返回 `np.ndarray`

形状和 dtype 库全都知道，没有理由让调用方 `np.array(output[0], np.float32).reshape(...)`。C++ 侧从 readback heap 按 desc 构造好 numpy 数组返回；`itemSize` 按 dtype 决定，一个 UINT8 输出张量读 N 字节而不是 4N。

### 4.3 `Expression` 的属性与可哈希

`.shape`（tuple）、`.strides`、`.dtype`（numpy dtype）、`.size`（元素数）、`.desc`（完整 `TensorDesc`）；`Buffer` 用同一套（§4.10）。`__repr__` 输出 `<dml.Expression float16 [1, 64, 512, 512]>`。没有默认构造：`Expression::Impl()` 为空指针的对象碰一下就段错误。

### 4.4 `graph.input(...)` 与 `TensorDesc` 的收敛

```python
graph.input(sizes=None, dtype=np.float32, *, owned=False, strides=None, desc=None, name=None)
```

常用路径不碰 `TensorDesc` 和 `TensorFlags`；要精确控制（`total_tensor_size_in_bytes`、`guaranteed_base_offset_alignment`）就传完整 `desc=`，此时其余参数一律非法（`TypeError`，`sizes` 也不例外）。desc 里已经写了一遍的东西再收一遍，就得回答「哪个赢」，那正是 §4.1 消灭掉的那类隐式契约。`TensorDesc` 本身是一个构造函数加关键字默认值，不靠实参类型消歧的重载。

`TensorPolicy` 只对单个 `TensorDesc` 生效，图内部生成的中间张量走 `Graph` 的构造参数，所以 `dml.Graph(device, tensor_policy=...)` 是 `InterleavedChannel` 真正可用的前提。

### 4.5 算子签名

**位置参数只留张量，其余全部 kw-only**（pybind11 的 `py::kw_only()`）。上游 sample 里的 `dml.Device(True, True)`、`mean_variance_normalization(conv4, scale, bias, [0, 2, 3], 1, 1, 0.000009999999747378752, ...)` 是这条规则的来源。

**默认值与参数顺序**照 DirectMLX：`epsilon=1e-5`、`normalize_variance` / `normalize_mean` 默认 `True`、GRU 的 `direction` 默认 `FORWARD`、`output_options` 默认 `Both`。必需参数不排在可选参数之后。`mean_variance_normalization` 的 `normalize_mean` 和 `average_pooling` 的 `dilations` 在 C++ 里是用 `DML_TARGET_VERSION` 条件编译插在参数表中间的，绑定位置与 C++ 逐字一致。

**命名对齐 C++**：`upsample_2d`（`Upsample2D` 一个词）、`join(inputs=)`、`GRUOutputOptions`、`CompiledOperator`、`activation_softmax`。

**`reinterpret(x, sizes, strides=None, dtype=None)`**，`dtype=None` 表示不变，因为绝大多数 reinterpret 不改 dtype。

**`FusedActivation` 的 18 个静态工厂**：`dml.FusedActivation.relu()`、`leaky_relu(alpha=0.01)`、`hard_sigmoid(alpha=0.2, beta=0.5)` 等，默认值照 DirectMLX 逐个搬。纯 1:1 补全，顺带把 `FusedActivation(OperatorType.CONVOLUTION)` 这种非法组合排除在外。落在包装层，一个工厂三行。

**docstring 写在包装层**，Args / Returns / Raises 齐全；`_core` 的 docstring 只有一句话。参数含义不能望文生义的算子（`local_response_normalization` 的 `alpha` / `beta` / `bias`）在包装层再包一层只为放 docstring。

### 4.6 多输出算子返回 namedtuple，未请求的输出是 `None`

`max_pooling(...)` 返回 `MaxPoolingOutputs(values, indices)`；`output_indices=False` 时 `indices is None`，而不是 DirectMLX 返回的那个碰一下就段错误的空 `Expression`。GRU 同理。namedtuple 用 `typing.NamedTuple` 定义在包装层，`_core` 返回普通元组。

### 4.7 执行模型：`initialize` / `dispatch` 挂在 `CompiledOperator` 上

上游的 `Device::Compute` 每次调用都重跑一遍初始化，且 persistent resource 挂在 `Device` 上：初始化第二个算子会覆盖第一个的内容。现在 persistent resource 归编译好的算子（连同它的 `ResourceAllocator` 引用，否则 Python 先销毁 `Device` 再销毁算子时段错误），资源在谁身上，方法就在谁身上。实测（RX 6800，`samples/sdxl` 解码，稳态）：

| 分辨率 | 每次 initialize + dispatch | 只 dispatch | |
| --- | --- | --- | --- |
| 512x512 | 0.33 s | 0.16 s | 2.1x |
| 1024x1024 | 1.09 s | 0.70 s | 1.6x |

大头是不再重跑初始化，不是权重常驻：只切换 `OWNED_BY_DML` 标志做对照，收益只占一两成。初始化那一项跟图的规模走，权重那一项跟权重字节数走。

契约：

- `OWNED_BY_DML` 张量的数据在 `initialize` 时被读走，之后换数据要重新 `initialize`，且要给全所有 owned 输入。
- **图里有 owned 输入而没 initialize 就 dispatch，抛错并点名缺哪些**。自动路径只有一条：`compile` 时每个 owned 输入都是 `constant`（包括一个都没有），初始化在 `compile` 里做掉（§4.11）；有权重而忘了给，是错误不是默认值。
- 同步模型：每次 dispatch 在 `WaitForQueueToComplete` 上阻塞。异步与流水线是另一个议题，`Buffer`（§4.10）是它的前提。

### 4.8 打包成 package，发类型存根

`directml/` 包内放 `_core.pyd`，`__init__.py` 是分层原则里 Python 侧的落点：re-export `_core` 的类和枚举，再放签名整形的包装函数、namedtuple、`FusedActivation` 工厂、dtype 对照表和校验报错。包装层是普通 Python，签名和 docstring 自文档；`py.typed` 加手写的 `_core.pyi` 覆盖扩展里剩下的类，并把挂上去的成员照实声明。

包装层给 `_core` 的类加东西只用一种手法：**在 import 时把方法和属性挂到 `_core` 的类上**（`Expression.shape`、`Graph.__init__` / `input` / `constant` / `compile`、`CompiledOperator.dispatch`、`TensorDesc.__init__`、`Buffer.__init__`），不做 Python 子类。实例都在 C++ 侧创建（`Expression` 来自算子、`Buffer` 来自 dispatch、`TensorDesc` 来自 `expr.desc`），子类只能覆盖用户自己构造的那些，库交回来的仍是基类，`isinstance(expr.desc, dml.TensorDesc)` 就会是 False。

`_core` 里 `py::class_` 注册整体在 `module.def` 之前，类与类之间按依赖排序：pybind11 在 `def` 的那一刻渲染签名字符串，未注册的类型退回 C++ 原始名。

`__version__` 从包元数据读。

### 4.9 `Expression` 的算术运算符

`+ - * / %` 和一元 `-` 来自 DirectMLX 的 C++ 重载：两个 Expression 之间是逐元素算子节点；float 标量骑在 identity 的 scale-bias 上（`x * 0.5` 是一个 `ELEMENT_WISE_IDENTITY`，不产生常量张量，也不占输入槽位）。三处不照单全收：

- **`float / x` 修正**。DirectMLX 写成 `Recip(x, {scale=a})`，但逐元素算子的 scale-bias 作用在**输入**上，算出来是 `1/(ax)` 而非 `a/x`。绑定改为先 `Recip` 再用 identity 乘回 `a`。
- **`%` 取 floored 语义**。Python 的 `-7 % 5 == 3`；DirectMLX 的 `operator%` 选了 `ModulusTruncate`（即 C 的 fmod），绑定换成 `ModulusFloor`。
- **不提供 in-place 形式**。`py::self += py::self` 会原地改写 C++ 节点，引用同一 Python 对象的所有别名一起变，hash 和 §4.1 的 dict 绑定身份随之失效。不定义 `__iadd__`，Python 自动退化为 `x = x + y`，只重绑一个名字。

逐元素二元运算（运算符和 `add` / `subtract` / `multiply` / `divide`）在**写下的那一行**校验两个操作数的 shape 和 dtype 一致，不一致抛 `ValueError` 并把两个 `Expression` 的 repr 都打出来。DirectML 到 `compile` 才拒绝，而且只给一个不带节点名的 `E_INVALIDARG`。float 操作数不校验。

两条红线：

- **比较运算符永远不构图**。`__eq__` / `__hash__` 按节点身份实现，numpy 风格的逐元素 `==` 与之不可共存；`<`、`>` 等一并不做，避免「一半按身份一半构图」的割裂。
- **不做隐式广播**。两个 Expression 的 shape 必须一致，广播由调用方用 `dml.broadcast` 显式表达（§4.12）。

### 4.10 `dml.Buffer`：GPU 常驻张量

把 `DML_BUFFER_BINDING` 只映射成 ndarray 会丢掉「数据在哪」这个维度：输入永远从 CPU 上传，输出永远经 readback 堆回到 CPU。而单个 D3D12 buffer 4 GiB 的上限（§5）**强迫**大模型拆图（sdxl 的 UNet 是两张），于是每个图边界都是一次 PCIe 往返，采样循环里每一步都付。`Buffer` 就是概念表里那个 `ID3D12Resource`，不是新抽象。

- `op(inputs, readback=False)` 给每个输出分配一块 DEFAULT 堆资源，直接绑成输出，返回 `Buffer` 而不是 ndarray。
- 绑定字典的值可以是 `Buffer`：直接绑定它的资源，不上传。dtype 必须与张量**完全一致**（GPU 上没有转换可做），字节数不得少于张量的 `TotalTensorSizeInBytes`，必须属于同一个 `Device`。`initialize` 和 `constant()` 同样收。
- `dml.Buffer(device, array, dtype=None)` 显式上传；`buffer.numpy()` 显式回读。**没有 `__array__`**：`np.asarray(buffer)` 悄悄做一次 GPU→CPU 拷贝是隐式传输，正是这一节要消灭的东西。
- `.shape` / `.strides` / `.dtype` / `.desc` / `.nbytes` 与 `Expression` 同一套属性。
- 生命周期：`Buffer` 持有 `Device` 的 `shared_ptr`，可以比 graph 和 op 都活得久。

### 4.11 `graph.constant(array)`

`owned=True` 把「值在构图时已知」和「DML 拥有它」绑在一起，多出一个阶段：真实用法里每个 owned 输入的数组在声明那一刻就在手上，用户只好自己攒一个 `weights` 字典带到 `initialize`，编译完再手动 `clear()`。这层进库：

```python
w = graph.constant(array, dtype=None, *, sizes=None, name=None)
```

- 声明一个 `OWNED_BY_DML` 输入，graph 记下**数组的引用**（不是拷贝）。`dtype` 默认取数组自己的 dtype（数组就在手边，和 `input()` 默认 float32 的理由不同）；cast 规则与 `dispatch` 同一张表，在 `constant()` 这一行就校验，报错指向声明处而不是 `compile`。`sizes` 允许用同元素数的另一个形状来看这块数据。
- `compile()` 把 constants 交给 op，graph 随即放手。若 owned 输入全是 constant（包括一个都没有），`compile` 就地 `initialize`，转换逐张量进行，op 也随即放手；库始终不保留拷贝。否则 constants 在 op 上等到 `initialize(weights)`，届时 `weights` 里没写的 constant 从记录里补，写了的以 `weights` 为准。
- 首次初始化之后记录清空；**再次 `initialize` 必须给全所有 owned 输入，constant 也不例外**。这是 §4.7「换数据要重新 initialize」的直接推论。
- graph 在 `compile` 后不再持有 constants。用户的 `Model` 若把 `self.graph` 一直挂着，也不会把 5 GiB 的 CPU 数组一起挂到进程结束。（一张 graph 本来也只能 `compile` 一次，见 §5。）

`owned=True` 保留给数据在构图时还没有的情形。

### 4.12 `dml.broadcast(x, shape)`

零步长视图是纯粹的张量描述符机制，没有任何模型语义，numpy 的广播规则又是确定的，而每个写 `x * scale` 的用户都要抄一遍那 20 行。库拒绝隐式广播是对的，提供显式的 `broadcast` 与之不冲突：它就是一个 `reinterpret`，不加算子、不拷贝。规则照 numpy：从右对齐，缺失的前导轴和长度为 1 的轴步长置 0，其余不一致报错；目标秩不得低于来源。来源已有 strides 的（比如转置视图）沿用其 strides，不重新按 packed 算。

### 4.13 输入的 `name=`

持有 `Expression` 句柄的对象和调用它的对象往往不是同一个，以 `Expression` 为键是正确的**原语**，不是终端用户的自然形态；没有名字，用户层会把位置绑定又造回来（`run(*values)` 按列表顺序 zip 成 dict）。所以 `graph.input(..., name=)` 和 `graph.constant(..., name=)` 接受一个 graph 内唯一的名字，绑定字典的 key 可以是 `Expression` 或名字，两者混用也行，同一个输入绑两次报错。报错信息里带名字：`input 0 'latent' (float16 [1, 4, 128, 128])`。不给名字什么都不变。

### 4.14 算子覆盖

DirectMLX 为推理包装的算子全部绑上，跳过的是训练用的 `*Grad`（`BatchNormalizationGrad`、`SliceGrad`、`RoiAlignGrad` 等）、`ConvolutionInteger` / `QuantizedLinearConvolution`，以及 FL 6.3 的 `Dequantize`（变长的量化参数张量加一个 `DML_QUANTIZATION_TYPE`，签名本身还要再设计一轮）。

几个命名和形状上的决定：

- **`dml.where(condition, a, b)`** 是 DirectMLX 的 `If`，`if` 是 Python 关键字。语义与 `np.where` 一致，名字就取后者的。
- **`abs` / `max` / `min` / `pow` / `round` 遮蔽内置名**，但只在 `dml.` 下，和 `np.abs` 一样（`slice` 早就这样了）。为了躲开而改名（`maximum`、`elementwise_max`）比遮蔽更糟：C++ 叫什么这里就叫什么。注意 `dml.max(a, b)` 是逐元素的两操作数算子，不是归约；归约是 `dml.reduce(x, function=ReduceFunction.MAX)`。
- **一元算子的 `scale_bias=(scale, bias)` 是一个元组**，不是两个参数、也不是一个类。DirectML 把它折进对输入的读取，算的是 `f(input * scale + bias)`；这两个数要么一起给要么一起不给，元组正好说明这件事。
- **`output_dtype` 只有 uint8 和 uint32 合法**（比较、`is_nan`、`is_infinity`、`bit_count`），别的类型 DirectML 直接 `E_INVALIDARG`；逻辑算子的输入也只吃这两种。库不再复述这张表，错误由 DirectML 报。
- **`fill_value_constant` / `fill_value_sequence` 的第一个位置参数是 graph**，与 DirectMLX 一致：它们没有输入张量，节点总得挂在某张图上。值在包装层用 numpy 转成张量类型的 8 个字节交给 `_core`，类型表在包装层，C++ 只做一次 memcpy；装不下的值（`value=300, dtype=uint8`）在这一步就报错。
- **`top_k` 返回 `TopKOutputs(values, indices)`**，复数与 `max_pooling` 对齐；DirectMLX 这里写的是 `value` / `index`，是它自己的不一致。
- **逐元素两操作数算子一律走 §4.9 的形状与类型检查**，包括 `max`、张量指数的 `pow`、以及 `where` 的 `a` / `b`（`condition` 只查形状）。报错里带调用名：`max(<...>, <...>): shapes differ`。

## 5. 硬约束

不属于 API 设计，但设计绕着它们走。

- **单个 D3D12 buffer 到 4 GiB 就到头了**，再大不是返回 `E_OUTOFMEMORY`，而是直接 `DXGI_ERROR_DEVICE_REMOVED`。阈值实测很干净：一张带 3.75 GiB 权重的图能初始化，4.00 GiB 的不能，且和显存余量无关。DirectML 把一个模型的 `OWNED_BY_DML` 权重折进**单个** persistent resource，所以**任何超过 4 GiB 权重的模型都建不成一张图**；`samples/sdxl` 的 UNet 半精度 4.78 GiB，只能在 mid block 处劈成两张图跑，两张之间靠 `Buffer` 传递（§4.10）。库自己的缓冲区增长（`util.h` 的 `GrowBufferSize`）过了 256 MiB 改按固定步长，避免一个 2.1 GiB 的请求被翻倍成 4 GiB 主动撞墙。
- **执行是同步的**。每次 `initialize` / `dispatch` / `Buffer.numpy()` 都在 fence 上等 GPU 完成。设备被移除时，是在这个等待点问 `GetDeviceRemovedReason`，所以错误会归到真正引起它的那次 dispatch 上，而不是之后某个无关调用。
- **一张 graph 只能 `compile` 一次**。DirectMLX 的 `Graph::Compile` 第二次调用被 DirectML 以 `E_INVALIDARG` 拒绝。要两个算子就建两张图。
- **`FILL_VALUE_SEQUENCE` 作为一张图的输出会让 DirectML 的图编译器直接 AV**，进程崩溃，不是一个 HRESULT。同一个节点喂给别的算子就没事，`FILL_VALUE_CONSTANT` 单独作输出也没事。包装层的 docstring 写明了这条，库不拦。
- **绑定一个没有任何表达式消费的图输入**会在 dispatch 时破坏设备（之后的 `CreateOperator` 失败）。这是 DirectML 侧的行为，库目前不拦。

## 6. 已拍板的争议项

1. **`Graph`**，不是 `GraphBuilder`。与 `dml::Graph` 名字对齐是一贯的判据，`compile()` 方法已把 builder 语义说清楚。
2. **枚举不收字符串**（`mode="nearest"` 这种 torch 风格）。枚举可 grep、可补全，字符串是又一套要维护的对照表。
3. **`graph.input` 用 `owned=True` 布尔**，不留 `flags=`。`TensorFlags` 实际只有 `NONE` 和 `OWNED_BY_DML` 两个值；为假想的第三个 flag 预留接口，正是本文档在别处反对的做法。
