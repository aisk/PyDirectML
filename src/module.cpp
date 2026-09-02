//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

PYBIND11_MODULE(_core, module)
{
    module.doc() = "C++ core of the directml package. Import directml, not this.";

    // Enumerations
    //
    py::enum_<DML_TENSOR_DATA_TYPE>(module, "TensorDataType")
        .value("UNKNOWN", DML_TENSOR_DATA_TYPE_UNKNOWN)
        .value("FLOAT32", DML_TENSOR_DATA_TYPE_FLOAT32)
        .value("FLOAT16", DML_TENSOR_DATA_TYPE_FLOAT16)
        .value("UINT32", DML_TENSOR_DATA_TYPE_UINT32)
        .value("UINT16", DML_TENSOR_DATA_TYPE_UINT16)
        .value("UINT8", DML_TENSOR_DATA_TYPE_UINT8)
        .value("INT32", DML_TENSOR_DATA_TYPE_INT32)
        .value("INT16", DML_TENSOR_DATA_TYPE_INT16)
        .value("INT8", DML_TENSOR_DATA_TYPE_INT8)
        .value("FLOAT64", DML_TENSOR_DATA_TYPE_FLOAT64)
        .value("UINT64", DML_TENSOR_DATA_TYPE_UINT64)
        .value("INT64", DML_TENSOR_DATA_TYPE_INT64);

    py::enum_<DML_TENSOR_FLAGS>(module, "TensorFlags")
        .value("NONE", DML_TENSOR_FLAG_NONE)
        .value("OWNED_BY_DML", DML_TENSOR_FLAG_OWNED_BY_DML);

    py::enum_<DML_MATRIX_TRANSFORM>(module, "MatrixTransform")
        .value("NONE", DML_MATRIX_TRANSFORM_NONE)
        .value("TRANSPOSE", DML_MATRIX_TRANSFORM_TRANSPOSE);

    py::enum_<DML_RECURRENT_NETWORK_DIRECTION>(module, "RecurrentNetworkDirection")
        .value("FORWARD", DML_RECURRENT_NETWORK_DIRECTION_FORWARD)
        .value("BACKWARD", DML_RECURRENT_NETWORK_DIRECTION_BACKWARD)
        .value("BIDIRECTIONAL", DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL);

    py::enum_<dml::GRUOutputOptions>(module, "GRUOutputOptions", py::arithmetic())
        .value("Both", dml::GRUOutputOptions::Both)
        .value("Sequence", dml::GRUOutputOptions::Sequence)
        .value("Single", dml::GRUOutputOptions::Single);

    py::enum_<DML_CONVOLUTION_MODE>(module, "ConvolutionMode")
        .value("CONVOLUTION", DML_CONVOLUTION_MODE_CONVOLUTION)
        .value("CROSS_CORRELATION", DML_CONVOLUTION_MODE_CROSS_CORRELATION);

    py::enum_<DML_CONVOLUTION_DIRECTION>(module, "ConvolutionDirection")
        .value("FORWARD", DML_CONVOLUTION_DIRECTION_FORWARD)
        .value("BACKWARD", DML_CONVOLUTION_DIRECTION_BACKWARD);

    py::enum_<DML_INTERPOLATION_MODE>(module, "InterpolationMode")
        .value("NEAREST_NEIGHBOR", DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR)
        .value("LINEAR", DML_INTERPOLATION_MODE_LINEAR);

    py::enum_<DML_PADDING_MODE>(module, "PaddingMode")
        .value("CONSTANT", DML_PADDING_MODE_CONSTANT)
        .value("EDGE", DML_PADDING_MODE_EDGE)
        .value("REFLECTION", DML_PADDING_MODE_REFLECTION);

    py::enum_<DML_EXECUTION_FLAGS>(module, "ExecutionFlags", py::arithmetic())
        .value("NONE", DML_EXECUTION_FLAG_NONE)
        .value("ALLOW_HALF_PRECISION_COMPUTATION", DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION)
        .value("DISABLE_META_COMMANDS", DML_EXECUTION_FLAG_DISABLE_META_COMMANDS)
        .value("DESCRIPTORS_VOLATILE", DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE);

    py::enum_<DML_OPERATOR_TYPE>(module, "OperatorType")
        .value("INVALID", DML_OPERATOR_INVALID)
        .value("ELEMENT_WISE_IDENTITY", DML_OPERATOR_ELEMENT_WISE_IDENTITY)
        .value("ELEMENT_WISE_ABS", DML_OPERATOR_ELEMENT_WISE_ABS)
        .value("ELEMENT_WISE_ACOS", DML_OPERATOR_ELEMENT_WISE_ACOS)
        .value("ELEMENT_WISE_ADD", DML_OPERATOR_ELEMENT_WISE_ADD)
        .value("ELEMENT_WISE_ASIN", DML_OPERATOR_ELEMENT_WISE_ASIN)
        .value("ELEMENT_WISE_ATAN", DML_OPERATOR_ELEMENT_WISE_ATAN)
        .value("ELEMENT_WISE_CEIL", DML_OPERATOR_ELEMENT_WISE_CEIL)
        .value("ELEMENT_WISE_CLIP", DML_OPERATOR_ELEMENT_WISE_CLIP)
        .value("ELEMENT_WISE_COS", DML_OPERATOR_ELEMENT_WISE_COS)
        .value("ELEMENT_WISE_DIVIDE", DML_OPERATOR_ELEMENT_WISE_DIVIDE)
        .value("ELEMENT_WISE_EXP", DML_OPERATOR_ELEMENT_WISE_EXP)
        .value("ELEMENT_WISE_FLOOR", DML_OPERATOR_ELEMENT_WISE_FLOOR)
        .value("ELEMENT_WISE_LOG", DML_OPERATOR_ELEMENT_WISE_LOG)
        .value("ELEMENT_WISE_LOGICAL_AND", DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND)
        .value("ELEMENT_WISE_LOGICAL_EQUALS", DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS)
        .value("ELEMENT_WISE_LOGICAL_GREATER_THAN", DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN)
        .value("ELEMENT_WISE_LOGICAL_LESS_THAN", DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN)
        .value("ELEMENT_WISE_LOGICAL_NOT", DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT)
        .value("ELEMENT_WISE_LOGICAL_OR", DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR)
        .value("ELEMENT_WISE_LOGICAL_XOR", DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR)
        .value("ELEMENT_WISE_MAX", DML_OPERATOR_ELEMENT_WISE_MAX)
        .value("ELEMENT_WISE_MEAN", DML_OPERATOR_ELEMENT_WISE_MEAN)
        .value("ELEMENT_WISE_MIN", DML_OPERATOR_ELEMENT_WISE_MIN)
        .value("ELEMENT_WISE_MULTIPLY", DML_OPERATOR_ELEMENT_WISE_MULTIPLY)
        .value("ELEMENT_WISE_POW", DML_OPERATOR_ELEMENT_WISE_POW)
        .value("ELEMENT_WISE_CONSTANT_POW", DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW)
        .value("ELEMENT_WISE_RECIP", DML_OPERATOR_ELEMENT_WISE_RECIP)
        .value("ELEMENT_WISE_SIN", DML_OPERATOR_ELEMENT_WISE_SIN)
        .value("ELEMENT_WISE_SQRT", DML_OPERATOR_ELEMENT_WISE_SQRT)
        .value("ELEMENT_WISE_SUBTRACT", DML_OPERATOR_ELEMENT_WISE_SUBTRACT)
        .value("ELEMENT_WISE_TAN", DML_OPERATOR_ELEMENT_WISE_TAN)
        .value("ELEMENT_WISE_THRESHOLD", DML_OPERATOR_ELEMENT_WISE_THRESHOLD)
        .value("ELEMENT_WISE_QUANTIZE_LINEAR", DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR)
        .value("ELEMENT_WISE_DEQUANTIZE_LINEAR", DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR)
        .value("ACTIVATION_ELU", DML_OPERATOR_ACTIVATION_ELU)
        .value("ACTIVATION_HARDMAX", DML_OPERATOR_ACTIVATION_HARDMAX)
        .value("ACTIVATION_HARD_SIGMOID", DML_OPERATOR_ACTIVATION_HARD_SIGMOID)
        .value("ACTIVATION_IDENTITY", DML_OPERATOR_ACTIVATION_IDENTITY)
        .value("ACTIVATION_LEAKY_RELU", DML_OPERATOR_ACTIVATION_LEAKY_RELU)
        .value("ACTIVATION_LINEAR", DML_OPERATOR_ACTIVATION_LINEAR)
        .value("ACTIVATION_LOG_SOFTMAX", DML_OPERATOR_ACTIVATION_LOG_SOFTMAX)
        .value("ACTIVATION_PARAMETERIZED_RELU", DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU)
        .value("ACTIVATION_PARAMETRIC_SOFTPLUS", DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS)
        .value("ACTIVATION_RELU", DML_OPERATOR_ACTIVATION_RELU)
        .value("ACTIVATION_SCALED_ELU", DML_OPERATOR_ACTIVATION_SCALED_ELU)
        .value("ACTIVATION_SCALED_TANH", DML_OPERATOR_ACTIVATION_SCALED_TANH)
        .value("ACTIVATION_SIGMOID", DML_OPERATOR_ACTIVATION_SIGMOID)
        .value("ACTIVATION_SOFTMAX", DML_OPERATOR_ACTIVATION_SOFTMAX)
        .value("ACTIVATION_SOFTPLUS", DML_OPERATOR_ACTIVATION_SOFTPLUS)
        .value("ACTIVATION_SOFTSIGN", DML_OPERATOR_ACTIVATION_SOFTSIGN)
        .value("ACTIVATION_TANH", DML_OPERATOR_ACTIVATION_TANH)
        .value("ACTIVATION_THRESHOLDED_RELU", DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU)
        .value("CONVOLUTION", DML_OPERATOR_CONVOLUTION)
        .value("GEMM", DML_OPERATOR_GEMM)
        .value("REDUCE", DML_OPERATOR_REDUCE)
        .value("AVERAGE_POOLING", DML_OPERATOR_AVERAGE_POOLING)
        .value("LP_POOLING", DML_OPERATOR_LP_POOLING)
        .value("MAX_POOLING", DML_OPERATOR_MAX_POOLING)
        .value("ROI_POOLING", DML_OPERATOR_ROI_POOLING)
        .value("SLICE", DML_OPERATOR_SLICE)
        .value("CAST", DML_OPERATOR_CAST)
        .value("SPLIT", DML_OPERATOR_SPLIT)
        .value("JOIN", DML_OPERATOR_JOIN)
        .value("PADDING", DML_OPERATOR_PADDING)
        .value("VALUE_SCALE_2D", DML_OPERATOR_VALUE_SCALE_2D)
        .value("UPSAMPLE_2D", DML_OPERATOR_UPSAMPLE_2D)
        .value("GATHER", DML_OPERATOR_GATHER)
        .value("SPACE_TO_DEPTH", DML_OPERATOR_SPACE_TO_DEPTH)
        .value("DEPTH_TO_SPACE", DML_OPERATOR_DEPTH_TO_SPACE)
        .value("TILE", DML_OPERATOR_TILE)
        .value("TOP_K", DML_OPERATOR_TOP_K)
        .value("BATCH_NORMALIZATION", DML_OPERATOR_BATCH_NORMALIZATION)
        .value("MEAN_VARIANCE_NORMALIZATION", DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION)
        .value("LOCAL_RESPONSE_NORMALIZATION", DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION)
        .value("LP_NORMALIZATION", DML_OPERATOR_LP_NORMALIZATION)
        .value("RNN", DML_OPERATOR_RNN)
        .value("LSTM", DML_OPERATOR_LSTM)
        .value("GRU", DML_OPERATOR_GRU)
        .value("ELEMENT_WISE_SIGN", DML_OPERATOR_ELEMENT_WISE_SIGN)
        .value("ELEMENT_WISE_IS_NAN", DML_OPERATOR_ELEMENT_WISE_IS_NAN)
        .value("ELEMENT_WISE_ERF", DML_OPERATOR_ELEMENT_WISE_ERF)
        .value("ELEMENT_WISE_SINH", DML_OPERATOR_ELEMENT_WISE_SINH)
        .value("ELEMENT_WISE_COSH", DML_OPERATOR_ELEMENT_WISE_COSH)
        .value("ELEMENT_WISE_TANH", DML_OPERATOR_ELEMENT_WISE_TANH)
        .value("ELEMENT_WISE_ASINH", DML_OPERATOR_ELEMENT_WISE_ASINH)
        .value("ELEMENT_WISE_ACOSH", DML_OPERATOR_ELEMENT_WISE_ACOSH)
        .value("ELEMENT_WISE_ATANH", DML_OPERATOR_ELEMENT_WISE_ATANH)
        .value("ELEMENT_WISE_IF", DML_OPERATOR_ELEMENT_WISE_IF)
        .value("ELEMENT_WISE_ADD1", DML_OPERATOR_ELEMENT_WISE_ADD1)
        .value("ACTIVATION_SHRINK", DML_OPERATOR_ACTIVATION_SHRINK)
        .value("MAX_POOLING1", DML_OPERATOR_MAX_POOLING1)
        .value("MAX_UNPOOLING", DML_OPERATOR_MAX_UNPOOLING)
        .value("DIAGONAL_MATRIX", DML_OPERATOR_DIAGONAL_MATRIX)
        .value("SCATTER", DML_OPERATOR_SCATTER)
        .value("ONE_HOT", DML_OPERATOR_ONE_HOT)
        .value("RESAMPLE", DML_OPERATOR_RESAMPLE)
        .value("ACTIVATION_CELU", DML_OPERATOR_ACTIVATION_CELU)
        .value("ACTIVATION_GELU", DML_OPERATOR_ACTIVATION_GELU);

    // Classes, registered before the module-level functions and ordered so that
    // every type is registered before another definition mentions it. pybind11
    // renders a definition's signature string at the moment of the def; a type
    // that is not registered yet shows up as its raw C++ name, which is not
    // valid Python and breaks the generated stubs.
    //
    py::class_<dml::TensorPolicy>(module, "TensorPolicy")
        .def_property_readonly_static("default", [](py::object) { return dml::TensorPolicy::Default(); })
        .def_property_readonly_static("interleaved_channel", [](py::object) { return dml::TensorPolicy::InterleavedChannel(); });

    py::class_<dml::TensorDesc>(module, "TensorDesc")
        // One constructor with keyword defaults, in place of four overloads
        // disambiguated by argument type. Everything past the sizes is detail,
        // so it is keyword-only.
        .def(py::init([](
            DML_TENSOR_DATA_TYPE dataType,
            dml::TensorDimensions sizes,
            DML_TENSOR_FLAGS flags,
            std::optional<dml::TensorDimensions> strides,
            std::optional<uint64_t> totalTensorSizeInBytes,
            uint32_t guaranteedBaseOffsetAlignment,
            std::optional<dml::TensorPolicy> tensorPolicy) {
                if (tensorPolicy && strides)
                {
                    throw std::invalid_argument(
                        "a tensor policy decides the strides; pass strides= or tensor_policy=, not both");
                }
                if (tensorPolicy)
                {
                    return dml::TensorDesc(dataType, flags, std::move(sizes), *tensorPolicy);
                }
                uint64_t total = totalTensorSizeInBytes ? *totalTensorSizeInBytes :
                    DMLCalcBufferTensorSize(
                        dataType,
                        static_cast<uint32_t>(sizes.size()),
                        sizes.data(),
                        strides ? strides->data() : nullptr);
                return dml::TensorDesc(
                    dataType,
                    flags,
                    std::move(sizes),
                    std::move(strides),
                    total,
                    guaranteedBaseOffsetAlignment);
            }),
            py::arg("data_type"),
            py::arg("sizes"),
            py::kw_only(),
            py::arg("flags") = DML_TENSOR_FLAG_NONE,
            py::arg("strides") = py::none(),
            py::arg("total_tensor_size_in_bytes") = py::none(),
            py::arg("guaranteed_base_offset_alignment") = 0,
            py::arg("tensor_policy") = py::none())
        .def_readonly("data_type", &dml::TensorDesc::dataType)
        .def_readonly("flags", &dml::TensorDesc::flags)
        .def_readonly("sizes", &dml::TensorDesc::sizes)
        .def_readonly("strides", &dml::TensorDesc::strides)
        .def_readonly("total_tensor_size_in_bytes", &dml::TensorDesc::totalTensorSizeInBytes)
        .def_readonly("guaranteed_base_offset_alignment", &dml::TensorDesc::guaranteedBaseOffsetAlignment)
        .def("__repr__",
            [](dml::TensorDesc const& tensorDesc) {
                return  "dml.TensorDesc of type " + std::to_string(tensorDesc.dataType) +
                        " and shape [" + UintVectorToString(tensorDesc.sizes) + ']' +
                        " with flags " + std::to_string(tensorDesc.flags);
            });

    py::class_<dml::Expression>(module, "Expression")
        .def_property_readonly("desc", &dml::Expression::GetOutputDesc,
            "The TensorDesc describing this expression's output.")
        .def_property_readonly("_node_id", [](dml::Expression const& self) {
            return reinterpret_cast<std::uintptr_t>(self.Impl());
            },
            "The node pointer as an integer: the exact identity value hash() is "
            "derived from. The wrapper layer matches dict keys against the input "
            "slots with it.")
        // Two Expressions are the same input if and only if they are the same
        // node, so identity by node pointer is what lets a dict keyed by
        // Expression replace bindings matched by position. The pointer is only
        // ever compared, never dereferenced, which is why a key stays usable
        // after its graph is destroyed.
        .def("__hash__", [](dml::Expression const& self) {
            return reinterpret_cast<std::uintptr_t>(self.Impl());
            })
        .def("__eq__", [](dml::Expression const& self, dml::Expression const& other) {
            return self.Impl() == other.Impl();
            }, py::is_operator())
        .def("__ne__", [](dml::Expression const& self, dml::Expression const& other) {
            return self.Impl() != other.Impl();
            }, py::is_operator())
        // The arithmetic operators are DirectMLX's: +, -, *, / build the
        // elementwise node, and a float operand rides on the scale-bias of an
        // identity. Two are corrected here rather than taken as-is, and the
        // in-place forms are dropped: += would mutate the C++ node behind
        // every Python reference at once, changing the hash and dict-binding
        // identity of each alias. Without __iadd__, Python rewrites x += y as
        // x = x + y, which rebinds one name and leaves the aliases alone.
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        // Python's % is floored: -7 % 5 == 3. DirectMLX's operator% picks
        // ModulusTruncate, which is C's fmod, so bind the floored operator
        // DirectML also has.
        .def("__mod__", [](dml::Expression const& a, dml::Expression const& b) {
            return dml::ModulusFloor(a, b);
            }, py::is_operator())
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        // DirectMLX writes a / x as Recip(x, scale=a), but an elementwise
        // scale-bias applies to the operator's input, so that computes
        // 1/(a*x). Scale the reciprocal instead.
        .def("__rtruediv__", [](dml::Expression const& self, float a) {
            return dml::Identity(dml::Recip(self), DML_SCALE_BIAS{ a, 0.0f });
            }, py::is_operator())
        .def(-py::self);

    py::class_<dml::FusedActivation>(module, "FusedActivation")
        .def(py::init<DML_OPERATOR_TYPE, float, float>(),
            py::arg("activation"),
            py::arg("param_1") = 0,
            py::arg("param_2") = 0);

    py::class_<pydml::Device, std::shared_ptr<pydml::Device>>(module, "Device")
        .def(py::init<bool, bool>(),
            py::kw_only(),
            py::arg("use_gpu") = true,
            py::arg("use_debug_layer") = false)
        .def("__repr__",
            [](pydml::Device const& device) {
                return "dml.Device on " + std::string(device.UseGpu() ? "GPU" : "CPU");
            });

    // A tensor on the GPU. Constructed from an array by the wrapper layer's
    // __init__, which converts to the desc's dtype first; handed out by
    // dispatch(readback=False); accepted wherever an array is.
    py::class_<pydml::Buffer, std::shared_ptr<pydml::Buffer>>(module, "Buffer")
        .def(py::init([](std::shared_ptr<pydml::Device> device, dml::TensorDesc desc, py::array array) {
            return device->Upload(std::move(desc), std::move(array));
            }),
            "Upload a C-contiguous array of desc's dtype into a new Buffer.",
            py::arg("device"),
            py::arg("desc"),
            py::arg("array"))
        .def_property_readonly("desc", [](pydml::Buffer const& self) { return self.desc; },
            "The TensorDesc this buffer is read through.")
        .def_property_readonly("device", [](pydml::Buffer const& self) { return self.device; },
            "The Device whose memory this is.")
        .def_property_readonly("nbytes", &pydml::Buffer::SizeInBytes,
            "Bytes of GPU memory the buffer holds.")
        .def("numpy", [](pydml::Buffer const& self) { return self.device->Download(self); },
            "Copy the buffer back to the host as a numpy array of its shape and dtype.");

    // The private surface of a compiled operator. The public initialize(),
    // dispatch() and __call__ live in the wrapper layer, which validates the
    // dict of inputs, separates the Buffers (bound in place) from the arrays
    // (converted one at a time as _initialize and _dispatch pull on the staged
    // iterable). py::dynamic_attr lets the wrapper cache its slot table on the
    // instance.
    py::class_<pydml::CompiledOperator>(module, "CompiledOperator", py::dynamic_attr())
        .def_property_readonly("temporary_size", [](pydml::CompiledOperator& self) {
            return self.op->GetBindingProperties().TemporaryResourceSize;
            },
            "Bytes of scratch memory one dispatch of this graph needs. Every "
            "intermediate tensor lives here, so this is the number that grows with "
            "the size of the input rather than with the size of the weights.")
        .def_property_readonly("persistent_size", [](pydml::CompiledOperator& self) {
            return self.op->GetBindingProperties().PersistentResourceSize;
            },
            "Bytes this graph keeps between dispatches: the OWNED_BY_DML tensors, "
            "in whatever layout the operators wanted them in.")
        .def_property_readonly("descriptor_count", [](pydml::CompiledOperator& self) {
            return self.op->GetBindingProperties().RequiredDescriptorCount;
            },
            "Descriptors one dispatch of this graph binds.")
        .def_property_readonly("initialized", [](pydml::CompiledOperator& self) {
            return self.initialized;
            },
            "Whether the OWNED_BY_DML inputs have been folded into the operator's "
            "persistent resource yet.")
        .def_property_readonly("_input_slots", [](pydml::CompiledOperator& self) {
            py::list slots;
            for (auto const& slot : self.inputs)
            {
                slots.append(py::make_tuple(slot.key, slot.owned, slot.desc));
            }
            return slots;
            },
            "One (node id, owned, TensorDesc) tuple per graph input, in index order.")
        .def("_initialize", [](pydml::CompiledOperator& self, pydml::Device::BufferMap const& buffers, py::iterable staged) {
            self.device->Initialize(self, buffers, staged);
            },
            py::arg("buffers"),
            py::arg("staged"))
        .def("_dispatch", [](pydml::CompiledOperator& self, pydml::Device::BufferMap const& buffers, py::iterable staged, bool readback) {
            return self.device->Dispatch(self, buffers, staged, readback);
            },
            py::arg("buffers"),
            py::arg("staged"),
            py::arg("readback"));

    // py::dynamic_attr for the wrapper's input names and constants, which live
    // on the graph until compile() hands them to the operator.
    py::class_<pydml::Graph>(module, "Graph", py::dynamic_attr())
        .def(py::init([](std::shared_ptr<pydml::Device> device, std::optional<dml::TensorPolicy> tensorPolicy) {
            return new pydml::Graph(std::move(device), tensorPolicy.value_or(dml::TensorPolicy::Default()));
            }),
            "A graph under construction. tensor_policy= decides the layout of the "
            "tensors the graph creates internally -- InterleavedChannel is only "
            "reachable through it.",
            py::arg("device"),
            py::kw_only(),
            py::arg("tensor_policy") = py::none())
        .def("_input", [](pydml::Graph& self, dml::TensorDesc desc) {
            return self.Input(static_cast<uint32_t>(self.slots.size()), std::move(desc));
            },
            "Add an input with the next free index; the public input() with its "
            "defaults lives in the wrapper layer.",
            py::arg("desc"))
        .def("_compile", [](pydml::Graph& self, std::vector<dml::Expression> outputs, DML_EXECUTION_FLAGS flags) {
            return new pydml::CompiledOperator(self, flags, outputs);
            },
            "Compile the graph; the public compile() with its docstring lives "
            "in the wrapper layer.",
            py::arg("outputs"),
            py::kw_only(),
            py::arg("flags") = DML_EXECUTION_FLAG_NONE);

    // Functions. Tensors are the only positional parameters; everything else is
    // keyword-only.
    //

    module.def("convolution", [](
        dml::Expression input,
        dml::Expression filter,
        dml::Optional<dml::Expression> bias,
        DML_CONVOLUTION_MODE mode,
        DML_CONVOLUTION_DIRECTION direction,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> dilations,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        std::vector<uint32_t> outputPadding,
        uint32_t groupCount,
        std::optional<dml::FusedActivation> fusedActivation,
        dml::TensorDimensions outputSizes) {
            return dml::Convolution(input, filter, bias, mode, direction, strides, dilations, startPadding, endPadding, outputPadding, groupCount, fusedActivation.value_or(dml::FusedActivation::None()), outputSizes);
        },
        "Create a builder of the convolution expression.",
        py::arg("input"),
        py::arg("filter"),
        py::arg("bias") = dml::NullOpt,
        py::kw_only(),
        py::arg("mode") = DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        py::arg("direction") = DML_CONVOLUTION_DIRECTION_FORWARD,
        py::arg("strides") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("output_padding") = std::vector<uint32_t>{},
        py::arg("group_count") = 1,
        py::arg("fused_activation") = py::none(),
        py::arg("output_sizes") = dml::TensorDimensions{});

    module.def("upsample_2d", [](
        dml::Expression input,
        std::pair<uint32_t, uint32_t> scaleSize,
        DML_INTERPOLATION_MODE interpolationMode) {
            return dml::Upsample2D(input, DML_SIZE_2D { scaleSize.first, scaleSize.second }, interpolationMode);
        },
        "Create a two-dimensional up-sample expression. scale_size is a "
        "(width, height) tuple.",
        py::arg("input"),
        py::kw_only(),
        py::arg("scale_size"),
        py::arg("interpolation_mode"));

    module.def("activation_relu", &dml::ActivationRelu, "Takes an input tensor and applies the function output = max(0, input) across its elements.",
        py::arg("input"));

    module.def("activation_sigmoid", &dml::ActivationSigmoid, "Takes an input tensor and applies the function output = 1 / (1 + exp(-input)) across its elements.",
        py::arg("input"));

    module.def("activation_identity", &dml::ActivationIdentity, "Takes an input tensor and return the tensor as an output.",
        py::arg("input"));

    module.def("add", [](dml::Expression a, dml::Expression b, std::optional<dml::FusedActivation> fusedActivation) {
            return dml::Add(a, b, fusedActivation.value_or(dml::FusedActivation::None()));
        },
        "Takes 2 input tensors and performs addition then returns the resulting tensor.",
        py::arg("a"),
        py::arg("b"),
        py::kw_only(),
        py::arg("fused_activation") = py::none());

    module.def("subtract", &dml::Subtract, "Takes 2 input tensors and performs subtraction then returns the resulting tensor.",
        py::arg("a"),
        py::arg("b"));

    module.def("activation_tanh", &dml::ActivationTanh, "Calculates the hyperbolic tangent of the given input tensor.",
        py::arg("input"));

    module.def("activation_gelu", &dml::ActivationGelu,
        "Gaussian error linear unit, f(input) = input * 0.5 * (1 + erf(input / sqrt(2))). "
        "This is the exact form, not the tanh approximation.",
        py::arg("input"));

    module.def("multiply", &dml::Multiply, "Takes 2 input tensors and performs multiplication then returns resulting tensor.",
        py::arg("a"),
        py::arg("b"));

    module.def("divide", &dml::Divide, "Takes 2 input tensors and performs division then returns the resulting tensor.",
        py::arg("a"),
        py::arg("b"));

    module.def("padding", [](
        dml::Expression input,
        DML_PADDING_MODE paddingMode,
        float paddingValue,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding) {
            return dml::Padding(input, paddingMode, paddingValue, startPadding, endPadding);
        },
        "Inflate the input at the edges, with constant, edge or reflection fill.",
        py::arg("input"),
        py::kw_only(),
        py::arg("padding_mode") = DML_PADDING_MODE_CONSTANT,
        py::arg("padding_value") = 0.0f,
        py::arg("start_padding"),
        py::arg("end_padding"));

    module.def("mean_variance_normalization", [](
        dml::Expression input,
        dml::Optional<dml::Expression> scale,
        dml::Optional<dml::Expression> bias,
        std::vector<uint32_t> axes,
        bool normalizeVariance,
        bool normalizeMean,
        float epsilon,
        std::optional<dml::FusedActivation> fusedActivation) {
            return dml::MeanVarianceNormalization(input, scale, bias, axes, normalizeVariance, normalizeMean, epsilon, fusedActivation.value_or(dml::FusedActivation::None()));
        }, "Normalize inputs using output = scale * (input - mean) / sqrt(variance + epsilon) + bias, where mean and variance are computed over the given axes.",
        py::arg("input"),
        py::arg("scale") = dml::NullOpt,
        py::arg("bias") = dml::NullOpt,
        py::kw_only(),
        py::arg("axes"),
        py::arg("normalize_variance") = true,
        py::arg("normalize_mean") = true,
        py::arg("epsilon") = 1e-5f,
        py::arg("fused_activation") = py::none());

    module.def("slice", [](
        dml::Expression input,
        std::vector<uint32_t> inputWindowOffsets,
        std::vector<uint32_t> inputWindowSizes,
        std::vector<int32_t> inputWindowStrides) {
            return dml::Slice(input, inputWindowOffsets, inputWindowSizes, inputWindowStrides);
        },
        "Produces a slice of the input tensor along multiple axes",
        py::arg("input"),
        py::kw_only(),
        py::arg("input_window_offsets"),
        py::arg("input_window_sizes"),
        py::arg("input_window_strides"));

    module.def("value_scale_2d", [](
        dml::Expression input,
        float scale,
        std::vector<float> bias) {
            return dml::ValueScale2D(input, scale, bias);
        },
        "Scales and bias the input image per pixel. output = input * scale + bias[C]",
        py::arg("input"),
        py::kw_only(),
        py::arg("scale"),
        py::arg("bias"));

    module.def("activation_linear", &dml::ActivationLinear, "f(input, alpha, beta) = alpha * input + beta",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha"),
        py::arg("beta"));

    module.def("batch_normalization", [](
        dml::Expression input,
        dml::Expression mean,
        dml::Expression variance,
        dml::Expression scale,
        dml::Expression bias,
        bool spatial,
        float epsilon,
        std::optional<dml::FusedActivation> fusedActivation) {
            return dml::BatchNormalization(input, mean, variance, scale, bias, spatial, epsilon, fusedActivation.value_or(dml::FusedActivation::None()));
        },
        "normalizes data per channel across all batches by subtracting the mean, dividing by the standard deviation, and adding a bias.",
        py::arg("input"),
        py::arg("mean"),
        py::arg("variance"),
        py::arg("scale"),
        py::arg("bias"),
        py::kw_only(),
        py::arg("spatial") = true,
        py::arg("epsilon") = 1e-5f,
        py::arg("fused_activation") = py::none());

    module.def("local_response_normalization", &dml::LocalResponseNormalization, "It normalizes over local input regions defined across the channels.",
        py::arg("input"),
        py::kw_only(),
        py::arg("cross_channel"),
        py::arg("local_size"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("bias"));

    module.def("gemm", [](
        dml::Expression a,
        dml::Expression b,
        dml::Optional<dml::Expression> c,
        DML_MATRIX_TRANSFORM transA,
        DML_MATRIX_TRANSFORM transB,
        float alpha,
        float beta,
        std::optional<dml::FusedActivation> fusedActivation) {
            return dml::Gemm(a, b, c, transA, transB, alpha, beta, fusedActivation.value_or(dml::FusedActivation::None()));
        },
        "Matrix product of two matrices",
        py::arg("a"),
        py::arg("b"),
        py::arg("c") = dml::NullOpt,
        py::kw_only(),
        py::arg("trans_a") = DML_MATRIX_TRANSFORM_NONE,
        py::arg("trans_b") = DML_MATRIX_TRANSFORM_NONE,
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 1.0f,
        py::arg("fused_activation") = py::none());

    module.def("average_pooling", [](
        dml::Expression input,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> windowSizes,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        std::vector<uint32_t> dilations,
        bool includePadding,
        dml::TensorDimensions outputSizes) {
            return dml::AveragePooling(input, strides, windowSizes, startPadding, endPadding, dilations, includePadding, outputSizes);
        },
        "Average all elements in each pool.",
        py::arg("input"),
        py::kw_only(),
        py::arg("strides"),
        py::arg("window_sizes"),
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("include_padding") = false,
        py::arg("output_sizes") = dml::TensorDimensions{});

    module.def("max_pooling", [](
        dml::Expression input,
        std::vector<uint32_t> windowSizes,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        std::vector<uint32_t> dilations,
        bool outputIndices) {
            auto outputs = dml::MaxPooling(input, windowSizes, strides, startPadding, endPadding, dilations, outputIndices);
            // The unrequested output is None, never an empty Expression that
            // dereferences null when touched. The wrapper layer turns the pair
            // into a namedtuple.
            return py::make_tuple(
                outputs.values,
                outputIndices ? py::cast(outputs.indices) : py::none());
        },
        "Max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths",
        py::arg("input"),
        py::kw_only(),
        py::arg("window_sizes"),
        py::arg("strides") = std::vector<uint32_t>{},
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("output_indices") = false);

    module.def("reinterpret", [](
        dml::Expression input,
        dml::TensorDimensions sizes,
        std::optional<dml::TensorStrides> strides,
        std::optional<DML_TENSOR_DATA_TYPE> dtype) {
            return dml::Reinterpret(
                input,
                dtype.value_or(input.GetOutputDesc().dataType),
                std::move(sizes),
                strides ? dml::Optional<dml::TensorStrides>(std::move(*strides)) : dml::NullOpt);
        },
        "Return tensor with a different view of the data, like a reinterpret cast "
        "using new dimensions that are element-count compatible. dtype=None keeps "
        "the input's type.",
        py::arg("input"),
        py::arg("sizes"),
        py::arg("strides") = py::none(),
        py::arg("dtype") = py::none());

    module.def("activation_softmax", [](dml::Expression input, std::vector<uint32_t> axes) {
            // An empty axis list selects the legacy operator, which normalizes along
            // the last dimension of a flattened 2-D view of the tensor.
            return axes.empty() ? dml::ActivationSoftmax(input) : dml::ActivationSoftmax(input, axes);
        },
        "Raise all elements to e, and divide each element by the sum over the given axes.",
        py::arg("input"),
        py::kw_only(),
        py::arg("axes") = std::vector<uint32_t>{});

    module.def("multihead_attention", [](
        dml::Expression query,
        dml::Expression key,
        dml::Expression value,
        uint32_t headCount,
        float scale) {
            return pydml::MultiHeadAttention(query, key, value, headCount, scale);
        },
        "Scaled dot-product attention over token tensors, as one operator rather "
        "than a gemm, a softmax and a gemm. The score matrix never becomes a "
        "tensor of its own, which is what makes this cheaper than writing it out.",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::kw_only(),
        py::arg("head_count"),
        py::arg("scale"));

    module.def("join", [](
        std::vector<dml::Expression> inputs,
        uint32_t axis) {
            return dml::Join(inputs, axis);
        },
        "Combine multiple tensors into large output tensor.",
        py::arg("inputs"),
        py::kw_only(),
        py::arg("axis"));

    module.def("gru", [](
        dml::Expression input,
        dml::Expression weight,
        dml::Expression recurrence,
        dml::Optional<dml::Expression> bias,
        dml::Optional<dml::Expression> hiddenInit,
        dml::Optional<dml::Expression> sequenceLengths,
        std::vector<dml::FusedActivation> activationDescs,
        DML_RECURRENT_NETWORK_DIRECTION direction,
        bool linearBeforeReset,
        dml::GRUOutputOptions outputOptions) {
            auto outputs = dml::GRU(input, weight, recurrence, bias, hiddenInit, sequenceLengths, activationDescs, direction, linearBeforeReset, outputOptions);
            // As with max_pooling: unrequested outputs are None, and the wrapper
            // layer shapes the pair into a namedtuple.
            bool sequence = outputOptions == dml::GRUOutputOptions::Both || outputOptions == dml::GRUOutputOptions::Sequence;
            bool single = outputOptions == dml::GRUOutputOptions::Both || outputOptions == dml::GRUOutputOptions::Single;
            return py::make_tuple(
                sequence ? py::cast(outputs.sequence) : py::none(),
                single ? py::cast(outputs.single) : py::none());
        },
        "Performs a one-layer gated recurrent unit (GRU) function on the input. This operator uses multiple gates to perform this layer. These gates are performed multiple times in a loop dictated by the sequence length dimension and the sequence_lengths argument.",
        py::arg("input"),
        py::arg("weight"),
        py::arg("recurrence"),
        py::arg("bias") = dml::NullOpt,
        py::arg("hidden_init") = dml::NullOpt,
        py::arg("sequence_lengths") = dml::NullOpt,
        py::kw_only(),
        py::arg("activation_descs"),
        py::arg("direction") = DML_RECURRENT_NETWORK_DIRECTION_FORWARD,
        py::arg("linear_before_reset") = true,
        py::arg("output_options") = dml::GRUOutputOptions::Both);

    module.def("gather", &dml::Gather, "Gathers elements from the input tensor along current axis, using indices tensor to remap indices.",
        py::arg("input"),
        py::arg("indices"),
        py::kw_only(),
        py::arg("axis"),
        py::arg("index_dimensions"));
}
