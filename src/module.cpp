//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

namespace
{
    // The optional linear transform an elementwise unary operator folds into
    // the read of its input, spelled in Python as a (scale, bias) pair.
    dml::Optional<DML_SCALE_BIAS> ToScaleBias(std::optional<std::pair<float, float>> const& scaleBias)
    {
        if (!scaleBias)
        {
            return dml::NullOpt;
        }
        return DML_SCALE_BIAS{ scaleBias->first, scaleBias->second };
    }

    // The value a fill operator writes, as the eight bytes of the tensor's own
    // type. The wrapper layer converts the Python number with numpy, which is
    // where the type table already lives; here it is only copied into place.
    DML_SCALAR_UNION ToScalarUnion(py::bytes const& value)
    {
        std::string bytes = value;
        if (bytes.size() != sizeof(DML_SCALAR_UNION::Bytes))
        {
            throw std::invalid_argument(
                "a fill value is " + std::to_string(sizeof(DML_SCALAR_UNION::Bytes)) +
                " bytes, not " + std::to_string(bytes.size()));
        }

        DML_SCALAR_UNION scalar = {};
        memcpy(scalar.Bytes, bytes.data(), sizeof(scalar.Bytes));
        return scalar;
    }
}

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

    py::enum_<DML_ROUNDING_MODE>(module, "RoundingMode")
        .value("HALVES_TO_NEAREST_EVEN", DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN)
        .value("TOWARD_ZERO", DML_ROUNDING_MODE_TOWARD_ZERO)
        .value("TOWARD_INFINITY", DML_ROUNDING_MODE_TOWARD_INFINITY);

    py::enum_<DML_IS_INFINITY_MODE>(module, "IsInfinityMode")
        .value("EITHER", DML_IS_INFINITY_MODE_EITHER)
        .value("POSITIVE", DML_IS_INFINITY_MODE_POSITIVE)
        .value("NEGATIVE", DML_IS_INFINITY_MODE_NEGATIVE);

    py::enum_<DML_AXIS_DIRECTION>(module, "AxisDirection")
        .value("INCREASING", DML_AXIS_DIRECTION_INCREASING)
        .value("DECREASING", DML_AXIS_DIRECTION_DECREASING);

    py::enum_<DML_DEPTH_SPACE_ORDER>(module, "DepthSpaceOrder")
        .value("DEPTH_COLUMN_ROW", DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW)
        .value("COLUMN_ROW_DEPTH", DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH);

    py::enum_<DML_REDUCE_FUNCTION>(module, "ReduceFunction")
        .value("ARGMAX", DML_REDUCE_FUNCTION_ARGMAX)
        .value("ARGMIN", DML_REDUCE_FUNCTION_ARGMIN)
        .value("AVERAGE", DML_REDUCE_FUNCTION_AVERAGE)
        .value("L1", DML_REDUCE_FUNCTION_L1)
        .value("L2", DML_REDUCE_FUNCTION_L2)
        .value("LOG_SUM", DML_REDUCE_FUNCTION_LOG_SUM)
        .value("LOG_SUM_EXP", DML_REDUCE_FUNCTION_LOG_SUM_EXP)
        .value("MAX", DML_REDUCE_FUNCTION_MAX)
        .value("MIN", DML_REDUCE_FUNCTION_MIN)
        .value("MULTIPLY", DML_REDUCE_FUNCTION_MULTIPLY)
        .value("SUM", DML_REDUCE_FUNCTION_SUM)
        .value("SUM_SQUARE", DML_REDUCE_FUNCTION_SUM_SQUARE);

    py::enum_<DML_RANDOM_GENERATOR_TYPE>(module, "RandomGeneratorType")
        .value("PHILOX_4X32_10", DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10);

    py::enum_<DML_QUANTIZATION_TYPE>(module, "QuantizationType")
        // NONE is the marker other descriptors use for "not quantized";
        // dequantize rejects it, so it is not one of the choices here.
        .value("SCALE", DML_QUANTIZATION_TYPE_SCALE)
        .value("SCALE_ZERO_POINT", DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT);

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

    // Classes, in dependency order: pybind11 renders a definition's signature
    // string at the moment of the def, and a type not yet registered shows up
    // as its raw C++ name.
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
        // Identity by node pointer is what makes an Expression a dict key. The
        // pointer is only ever compared, never dereferenced, so a key stays
        // usable after its graph is destroyed.
        .def("__hash__", [](dml::Expression const& self) {
            return reinterpret_cast<std::uintptr_t>(self.Impl());
            })
        .def("__eq__", [](dml::Expression const& self, dml::Expression const& other) {
            return self.Impl() == other.Impl();
            }, py::is_operator())
        .def("__ne__", [](dml::Expression const& self, dml::Expression const& other) {
            return self.Impl() != other.Impl();
            }, py::is_operator())
        // DirectMLX's arithmetic overloads, with the deviations docs/api-design.md
        // lists: no in-place forms, a floored %, and a corrected float / x.
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
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

    // The public initialize(), dispatch() and __call__ live in the wrapper
    // layer, which validates the dict of inputs and splits it into Buffers
    // (bound in place) and staged arrays (converted one at a time as
    // _initialize and _dispatch pull on the iterable). py::dynamic_attr lets
    // the wrapper keep its slot table and names on the instance.
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

    // py::dynamic_attr for the wrapper's input names and constants.
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
            return self.Input(std::move(desc));
            },
            "Add an input at the next free index; the public input() lives in the wrapper layer.",
            py::arg("desc"))
        .def("_compile", [](pydml::Graph& self, std::vector<dml::Expression> outputs, DML_EXECUTION_FLAGS flags) {
            return new pydml::CompiledOperator(self, flags, outputs);
            },
            "Compile the graph; the public compile() lives in the wrapper layer.",
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
        "Convolve the input with the filter, adding the bias per output channel if given.",
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
        "Upsample the last two axes by scale_size, a (width, height) tuple.",
        py::arg("input"),
        py::kw_only(),
        py::arg("scale_size"),
        py::arg("interpolation_mode"));

    module.def("activation_relu", &dml::ActivationRelu, "Elementwise max(0, input).",
        py::arg("input"));

    module.def("activation_sigmoid", &dml::ActivationSigmoid, "Elementwise 1 / (1 + exp(-input)).",
        py::arg("input"));

    module.def("activation_identity", &dml::ActivationIdentity, "A copy of the input, packed.",
        py::arg("input"));

    module.def("add", [](dml::Expression a, dml::Expression b, std::optional<dml::FusedActivation> fusedActivation) {
            return dml::Add(a, b, fusedActivation.value_or(dml::FusedActivation::None()));
        },
        "Elementwise a + b, optionally with a fused activation on the result.",
        py::arg("a"),
        py::arg("b"),
        py::kw_only(),
        py::arg("fused_activation") = py::none());

    module.def("subtract", &dml::Subtract, "Elementwise a - b.",
        py::arg("a"),
        py::arg("b"));

    module.def("activation_tanh", &dml::ActivationTanh, "Elementwise hyperbolic tangent.",
        py::arg("input"));

    module.def("activation_gelu", &dml::ActivationGelu,
        "Gaussian error linear unit, f(input) = input * 0.5 * (1 + erf(input / sqrt(2))). "
        "This is the exact form, not the tanh approximation.",
        py::arg("input"));

    module.def("multiply", &dml::Multiply, "Elementwise a * b.",
        py::arg("a"),
        py::arg("b"));

    module.def("divide", &dml::Divide, "Elementwise a / b.",
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
        "Pad the input at both ends of every axis, with constant, edge or reflection fill.",
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
        "A window of the input: offsets, sizes and strides per axis.",
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
        "output = input * scale + bias[channel], one bias per channel.",
        py::arg("input"),
        py::kw_only(),
        py::arg("scale"),
        py::arg("bias"));

    module.def("activation_linear", &dml::ActivationLinear, "Elementwise alpha * input + beta.",
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
        "output = scale * (input - mean) / sqrt(variance + epsilon) + bias, with per-channel statistics supplied as tensors.",
        py::arg("input"),
        py::arg("mean"),
        py::arg("variance"),
        py::arg("scale"),
        py::arg("bias"),
        py::kw_only(),
        py::arg("spatial") = true,
        py::arg("epsilon") = 1e-5f,
        py::arg("fused_activation") = py::none());

    module.def("local_response_normalization", &dml::LocalResponseNormalization, "Normalize each element by the energy of its neighbourhood; the wrapper layer documents the parameters.",
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
        "alpha * op(a) @ op(b) + beta * c over the last two axes, where op is the optional transpose.",
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
        "Average pooling over the trailing spatial axes.",
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
            // An unrequested output is None, never an empty Expression that
            // dereferences null when touched.
            return py::make_tuple(
                outputs.values,
                outputIndices ? py::cast(outputs.indices) : py::none());
        },
        "Max pooling over the trailing spatial axes; the wrapper layer shapes the outputs.",
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
        "View the same bytes through different sizes, strides or dtype; dtype=None keeps the input's.",
        py::arg("input"),
        py::arg("sizes"),
        py::arg("strides") = py::none(),
        py::arg("dtype") = py::none());

    module.def("activation_softmax", [](dml::Expression input, std::vector<uint32_t> axes) {
            // An empty axis list selects the legacy operator, which normalizes along
            // the last dimension of a flattened 2-D view of the tensor.
            return axes.empty() ? dml::ActivationSoftmax(input) : dml::ActivationSoftmax(input, axes);
        },
        "Softmax over the given axes; no axes selects the legacy operator over a flattened 2-D view.",
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
        "Concatenate the inputs along an axis.",
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
            // As with max_pooling: unrequested outputs are None.
            bool sequence = outputOptions == dml::GRUOutputOptions::Both || outputOptions == dml::GRUOutputOptions::Sequence;
            bool single = outputOptions == dml::GRUOutputOptions::Both || outputOptions == dml::GRUOutputOptions::Single;
            return py::make_tuple(
                sequence ? py::cast(outputs.sequence) : py::none(),
                single ? py::cast(outputs.single) : py::none());
        },
        "A one-layer gated recurrent unit over the sequence axis; the wrapper layer shapes the outputs.",
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

    module.def("gather", &dml::Gather, "Pick elements of the input along an axis by an indices tensor.",
        py::arg("input"),
        py::arg("indices"),
        py::kw_only(),
        py::arg("axis"),
        py::arg("index_dimensions"));

    // Elementwise unary operators. Every one of them takes the same optional
    // (scale, bias) pair, which DirectML folds into the read of the input: the
    // operator computes f(input * scale + bias) for one pass over the data.
    //
#define PYDML_UNARY(_name, _function, _doc)                                     \
    module.def(_name,                                                           \
        [](dml::Expression input, std::optional<std::pair<float, float>> scaleBias) { \
            return dml::_function(input, ToScaleBias(scaleBias));               \
        },                                                                      \
        _doc,                                                                   \
        py::arg("input"),                                                       \
        py::kw_only(),                                                          \
        py::arg("scale_bias") = py::none())

    PYDML_UNARY("identity", Identity, "A copy of the input.");
    PYDML_UNARY("abs", Abs, "Elementwise absolute value.");
    PYDML_UNARY("acos", ACos, "Elementwise arc cosine, in radians.");
    PYDML_UNARY("asin", ASin, "Elementwise arc sine, in radians.");
    PYDML_UNARY("atan", ATan, "Elementwise arc tangent, in radians.");
    PYDML_UNARY("ceil", Ceil, "Elementwise round towards positive infinity.");
    PYDML_UNARY("cos", Cos, "Elementwise cosine of an angle in radians.");
    PYDML_UNARY("exp", Exp, "Elementwise e ** input.");
    PYDML_UNARY("floor", Floor, "Elementwise round towards negative infinity.");
    PYDML_UNARY("log", Log, "Elementwise natural logarithm.");
    PYDML_UNARY("recip", Recip, "Elementwise 1 / input.");
    PYDML_UNARY("sin", Sin, "Elementwise sine of an angle in radians.");
    PYDML_UNARY("sqrt", Sqrt, "Elementwise square root.");
    PYDML_UNARY("tan", Tan, "Elementwise tangent of an angle in radians.");
    PYDML_UNARY("erf", Erf, "Elementwise Gauss error function.");
    PYDML_UNARY("sinh", Sinh, "Elementwise hyperbolic sine.");
    PYDML_UNARY("cosh", Cosh, "Elementwise hyperbolic cosine.");
    PYDML_UNARY("tanh", Tanh, "Elementwise hyperbolic tangent.");
    PYDML_UNARY("asinh", ASinh, "Elementwise inverse hyperbolic sine.");
    PYDML_UNARY("acosh", ACosh, "Elementwise inverse hyperbolic cosine.");
    PYDML_UNARY("atanh", ATanh, "Elementwise inverse hyperbolic tangent.");

#undef PYDML_UNARY

    // The unary operators with no scale and bias to fold.
    //
    module.def("sign", &dml::Sign, "Elementwise -1, 0 or 1 by the sign of the input.",
        py::arg("input"));

    module.def("negate", &dml::Negate, "Elementwise -input.",
        py::arg("input"));

    module.def("logical_not", &dml::LogicalNot, "Elementwise 1 where the input is zero, 0 elsewhere.",
        py::arg("input"));

    module.def("bit_not", &dml::BitNot, "Elementwise bitwise complement.",
        py::arg("input"));

    // The unary operators with parameters of their own.
    //
    module.def("clip", [](
        dml::Expression input,
        float min,
        float max,
        std::optional<std::pair<float, float>> scaleBias) {
            return dml::Clip(input, min, max, ToScaleBias(scaleBias));
        },
        "Elementwise clamp of the input to [min, max].",
        py::arg("input"),
        py::kw_only(),
        py::arg("min"),
        py::arg("max"),
        py::arg("scale_bias") = py::none());

    module.def("threshold", [](
        dml::Expression input,
        float min,
        std::optional<std::pair<float, float>> scaleBias) {
            return dml::Threshold(input, min, ToScaleBias(scaleBias));
        },
        "Elementwise max(input, min), the one-sided clip.",
        py::arg("input"),
        py::kw_only(),
        py::arg("min"),
        py::arg("scale_bias") = py::none());

    module.def("round", [](dml::Expression input, DML_ROUNDING_MODE roundingMode) {
            return dml::Round(input, roundingMode);
        },
        "Elementwise round to an integer value, by the given rule.",
        py::arg("input"),
        py::kw_only(),
        py::arg("rounding_mode") = DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN);

    module.def("is_nan", [](dml::Expression input, DML_TENSOR_DATA_TYPE outputDataType) {
            return dml::IsNaN(input, outputDataType);
        },
        "Elementwise 1 where the input is NaN, 0 elsewhere.",
        py::arg("input"),
        py::kw_only(),
        py::arg("output_dtype") = DML_TENSOR_DATA_TYPE_UINT8);

    module.def("is_infinity", [](
        dml::Expression input,
        DML_IS_INFINITY_MODE infinityMode,
        DML_TENSOR_DATA_TYPE outputDataType) {
            return dml::IsInfinity(input, infinityMode, outputDataType);
        },
        "Elementwise 1 where the input is an infinity of the given sign, 0 elsewhere.",
        py::arg("input"),
        py::kw_only(),
        py::arg("infinity_mode") = DML_IS_INFINITY_MODE_EITHER,
        py::arg("output_dtype") = DML_TENSOR_DATA_TYPE_UINT8);

    module.def("bit_count", [](dml::Expression input, DML_TENSOR_DATA_TYPE outputDataType) {
            return dml::BitCount(input, outputDataType);
        },
        "Elementwise count of the bits set in the input.",
        py::arg("input"),
        py::kw_only(),
        py::arg("output_dtype") = DML_TENSOR_DATA_TYPE_UINT8);

    module.def("cast", [](dml::Expression input, DML_TENSOR_DATA_TYPE targetDataType) {
            return dml::Cast(input, targetDataType);
        },
        "Convert the elements to another type, as a static_cast would.",
        py::arg("input"),
        py::kw_only(),
        py::arg("dtype"));

    // Elementwise binary operators. Both operands want the same shape and type;
    // the wrapper layer says which one is wrong when they differ. The logical
    // and bitwise ones read and write a predicate, so uint8 or uint32.
    //
#define PYDML_BINARY(_name, _function, _doc)                                    \
    module.def(_name, &dml::_function, _doc, py::arg("a"), py::arg("b"))

    PYDML_BINARY("max", Max, "Elementwise larger of a and b.");
    PYDML_BINARY("min", Min, "Elementwise smaller of a and b.");
    PYDML_BINARY("mean", Mean, "Elementwise (a + b) / 2.");
    PYDML_BINARY("atan_yx", ATanYX, "Elementwise arc tangent of a / b, in the quadrant the signs of both name.");
    PYDML_BINARY("difference_square", DifferenceSquare, "Elementwise (a - b) ** 2.");
    PYDML_BINARY("logical_and", LogicalAnd, "Elementwise 1 where both operands are non-zero, 0 elsewhere.");
    PYDML_BINARY("logical_or", LogicalOr, "Elementwise 1 where either operand is non-zero, 0 elsewhere.");
    PYDML_BINARY("logical_xor", LogicalXor, "Elementwise 1 where exactly one operand is non-zero, 0 elsewhere.");
    PYDML_BINARY("bit_and", BitAnd, "Elementwise bitwise and.");
    PYDML_BINARY("bit_or", BitOr, "Elementwise bitwise or.");
    PYDML_BINARY("bit_xor", BitXor, "Elementwise bitwise exclusive or.");
    PYDML_BINARY("bit_shift_left", BitShiftLeft, "Elementwise a shifted left by b bits.");
    PYDML_BINARY("bit_shift_right", BitShiftRight, "Elementwise a shifted right by b bits.");
    PYDML_BINARY("modulus_truncate", ModulusTruncate, "Elementwise remainder with the sign of a, as C's % has.");
    PYDML_BINARY("modulus_floor", ModulusFloor, "Elementwise remainder with the sign of b, as Python's % has.");

#undef PYDML_BINARY

    // The comparisons, which write their result as the type asked for rather
    // than the operands'. uint8 and uint32 are the only two DirectML accepts,
    // here and on is_nan, is_infinity and bit_count.
    //
#define PYDML_COMPARISON(_name, _function, _doc)                                \
    module.def(_name,                                                           \
        [](dml::Expression a, dml::Expression b, DML_TENSOR_DATA_TYPE outputDataType) { \
            return dml::_function(a, b, outputDataType);                        \
        },                                                                      \
        _doc,                                                                   \
        py::arg("a"),                                                           \
        py::arg("b"),                                                           \
        py::kw_only(),                                                          \
        py::arg("output_dtype") = DML_TENSOR_DATA_TYPE_UINT8)

    PYDML_COMPARISON("equals", Equals, "Elementwise 1 where a == b, 0 elsewhere.");
    PYDML_COMPARISON("greater_than", GreaterThan, "Elementwise 1 where a > b, 0 elsewhere.");
    PYDML_COMPARISON("greater_than_or_equal", GreaterThanOrEqual, "Elementwise 1 where a >= b, 0 elsewhere.");
    PYDML_COMPARISON("less_than", LessThan, "Elementwise 1 where a < b, 0 elsewhere.");
    PYDML_COMPARISON("less_than_or_equal", LessThanOrEqual, "Elementwise 1 where a <= b, 0 elsewhere.");

#undef PYDML_COMPARISON

    // Pow is the one elementwise operator with a scalar form of its own: a
    // constant exponent is a different DirectML operator, not a constant tensor.
    //
    module.def("pow", [](
        dml::Expression input,
        dml::Expression exponent,
        std::optional<std::pair<float, float>> scaleBias) {
            return dml::Pow(input, exponent, ToScaleBias(scaleBias));
        },
        "Elementwise input ** exponent.",
        py::arg("input"),
        py::arg("exponent"),
        py::kw_only(),
        py::arg("scale_bias") = py::none());

    module.def("pow", [](
        dml::Expression input,
        float exponent,
        std::optional<std::pair<float, float>> scaleBias) {
            return dml::Pow(input, exponent, ToScaleBias(scaleBias));
        },
        "Elementwise input ** exponent for a constant exponent.",
        py::arg("input"),
        py::arg("exponent"),
        py::kw_only(),
        py::arg("scale_bias") = py::none());

    module.def("where", [](dml::Expression condition, dml::Expression a, dml::Expression b) {
            return dml::If(condition, a, b);
        },
        "Elementwise a where the condition is non-zero, b elsewhere.",
        py::arg("condition"),
        py::arg("a"),
        py::arg("b"));

    // Activations. Every one of these is also a FusedActivation an operator can
    // apply to its own output; these are the standalone nodes.
    //
    module.def("activation_elu", &dml::ActivationElu,
        "Exponential linear unit, input where positive and alpha * (exp(input) - 1) elsewhere.",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 1.0f);

    module.def("activation_celu", &dml::ActivationCelu,
        "Continuously differentiable exponential linear unit.",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 1.0f);

    module.def("activation_hardmax", &dml::ActivationHardmax,
        "1 at the largest element of each row of a flattened 2-D view, 0 elsewhere.",
        py::arg("input"));

    module.def("activation_hard_sigmoid", &dml::ActivationHardSigmoid,
        "Elementwise clip(alpha * input + beta, 0, 1).",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 0.2f,
        py::arg("beta") = 0.5f);

    module.def("activation_leaky_relu", &dml::ActivationLeakyRelu,
        "Elementwise input where positive, alpha * input elsewhere.",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 0.01f);

    module.def("activation_log_softmax", &dml::ActivationLogSoftmax,
        "The logarithm of the softmax, over the last dimension of a flattened 2-D view.",
        py::arg("input"));

    module.def("activation_parameterized_relu", &dml::ActivationParameterizedRelu,
        "Leaky relu with the slope of the negative half read from a tensor, one per channel.",
        py::arg("input"),
        py::arg("slope"));

    module.def("activation_parametric_softplus", &dml::ActivationParametricSoftplus,
        "Elementwise alpha * log(1 + exp(beta * input)).",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha"),
        py::arg("beta"));

    module.def("activation_scaled_elu", &dml::ActivationScaledElu,
        "The elu scaled by gamma, the self-normalizing activation.",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 1.67326319217681884765625f,
        py::arg("gamma") = 1.05070102214813232421875f);

    module.def("activation_scaled_tanh", &dml::ActivationScaledTanh,
        "Elementwise alpha * tanh(beta * input).",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.5f);

    module.def("activation_shrink", &dml::ActivationShrink,
        "Elementwise zero within threshold of zero, and shifted towards it by bias outside.",
        py::arg("input"),
        py::kw_only(),
        py::arg("bias") = 0.0f,
        py::arg("threshold") = 0.5f);

    module.def("activation_softplus", &dml::ActivationSoftplus,
        "Elementwise log(1 + exp(steepness * input)) / steepness.",
        py::arg("input"),
        py::kw_only(),
        py::arg("steepness") = 1.0f);

    module.def("activation_softsign", &dml::ActivationSoftsign,
        "Elementwise input / (1 + abs(input)).",
        py::arg("input"));

    module.def("activation_thresholded_relu", &dml::ActivationThresholdedRelu,
        "Elementwise input where it is above alpha, 0 elsewhere.",
        py::arg("input"),
        py::kw_only(),
        py::arg("alpha") = 1.0f);

    // Shape and data movement.
    //
    module.def("split", [](
        dml::Expression input,
        uint32_t axis,
        std::vector<uint32_t> outputAxisSizes) {
            return dml::Split(input, axis, outputAxisSizes);
        },
        "Cut the input along an axis into one output per requested extent.",
        py::arg("input"),
        py::kw_only(),
        py::arg("axis"),
        py::arg("output_axis_sizes"));

    module.def("tile", [](dml::Expression input, std::vector<uint32_t> repeats) {
            return dml::Tile(input, repeats);
        },
        "Repeat the input repeats[i] times along axis i.",
        py::arg("input"),
        py::kw_only(),
        py::arg("repeats"));

    module.def("one_hot", [](
        dml::Expression indices,
        dml::Expression values,
        uint32_t outputLength,
        uint32_t axis) {
            return dml::OneHot(indices, values, outputLength, axis);
        },
        "Expand an axis of indices to output_length, filled with values[1] at the index and values[0] elsewhere.",
        py::arg("indices"),
        py::arg("values"),
        py::kw_only(),
        py::arg("output_length"),
        py::arg("axis"));

    module.def("top_k", [](
        dml::Expression input,
        uint32_t axis,
        uint32_t k,
        DML_AXIS_DIRECTION axisDirection) {
            auto outputs = dml::TopK(input, axis, k, axisDirection);
            return py::make_tuple(outputs.value, outputs.index);
        },
        "The k largest or smallest elements along an axis; the wrapper layer shapes the outputs.",
        py::arg("input"),
        py::kw_only(),
        py::arg("axis"),
        py::arg("k"),
        py::arg("axis_direction") = DML_AXIS_DIRECTION_DECREASING);

    module.def("gather_elements", [](
        dml::Expression input,
        dml::Expression indices,
        uint32_t axis) {
            return dml::GatherElements(input, indices, axis);
        },
        "Pick one element per index along an axis, the indices tensor's shape out.",
        py::arg("input"),
        py::arg("indices"),
        py::kw_only(),
        py::arg("axis"));

    module.def("gather_nd", [](
        dml::Expression input,
        dml::Expression indices,
        uint32_t inputDimensionCount,
        uint32_t indicesDimensionCount,
        uint32_t batchDimensionCount) {
            return dml::GatherND(input, indices, inputDimensionCount, indicesDimensionCount, batchDimensionCount);
        },
        "Pick slices of the input by coordinates in the last axis of the indices tensor.",
        py::arg("input"),
        py::arg("indices"),
        py::kw_only(),
        py::arg("input_dimension_count"),
        py::arg("indices_dimension_count"),
        py::arg("batch_dimension_count") = 0);

    module.def("scatter_elements", [](
        dml::Expression input,
        dml::Expression indices,
        dml::Expression updates,
        uint32_t axis) {
            return dml::ScatterElements(input, indices, updates, axis);
        },
        "A copy of the input with the updates written at the indices along an axis.",
        py::arg("input"),
        py::arg("indices"),
        py::arg("updates"),
        py::kw_only(),
        py::arg("axis"));

    module.def("scatter_nd", [](
        dml::Expression input,
        dml::Expression indices,
        dml::Expression updates,
        uint32_t inputDimensionCount,
        uint32_t indicesDimensionCount) {
            return dml::ScatterND(input, indices, updates, inputDimensionCount, indicesDimensionCount);
        },
        "A copy of the input with the updates written at the coordinates in the indices tensor.",
        py::arg("input"),
        py::arg("indices"),
        py::arg("updates"),
        py::kw_only(),
        py::arg("input_dimension_count"),
        py::arg("indices_dimension_count"));

    module.def("space_to_depth", [](
        dml::Expression input,
        uint32_t blockSize,
        DML_DEPTH_SPACE_ORDER order) {
            return dml::SpaceToDepth(input, blockSize, order);
        },
        "Move block_size x block_size spatial patches into the channel axis.",
        py::arg("input"),
        py::kw_only(),
        py::arg("block_size"),
        py::arg("order") = DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW);

    module.def("depth_to_space", [](
        dml::Expression input,
        uint32_t blockSize,
        DML_DEPTH_SPACE_ORDER order) {
            return dml::DepthToSpace(input, blockSize, order);
        },
        "Move channels out into block_size x block_size spatial patches.",
        py::arg("input"),
        py::kw_only(),
        py::arg("block_size"),
        py::arg("order") = DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW);

    module.def("reverse_subsequences", [](
        dml::Expression input,
        dml::Expression sequenceLengths,
        uint32_t axis) {
            return dml::ReverseSubsequences(input, sequenceLengths, axis);
        },
        "Reverse the first sequence_lengths[i] elements of each subsequence along an axis.",
        py::arg("input"),
        py::arg("sequence_lengths"),
        py::kw_only(),
        py::arg("axis"));

    module.def("resample", [](
        dml::Expression input,
        dml::TensorDimensions outputSizes,
        DML_INTERPOLATION_MODE mode,
        DML_AXIS_DIRECTION roundingDirection,
        std::vector<float> scales,
        std::vector<float> inputPixelOffsets,
        std::vector<float> outputPixelOffsets,
        bool antialiased) {
            return dml::Resample(input, std::move(outputSizes), mode, roundingDirection,
                scales, inputPixelOffsets, outputPixelOffsets, antialiased);
        },
        "Resample every axis to output_sizes; the wrapper layer documents the offsets.",
        py::arg("input"),
        py::kw_only(),
        py::arg("output_sizes"),
        py::arg("mode"),
        py::arg("rounding_direction") = DML_AXIS_DIRECTION_INCREASING,
        py::arg("scales") = std::vector<float>{},
        py::arg("input_pixel_offsets") = std::vector<float>{},
        py::arg("output_pixel_offsets") = std::vector<float>{},
        py::arg("antialiased") = false);

    module.def("fill_value_constant", [](
        pydml::Graph& graph,
        dml::TensorDimensions sizes,
        DML_TENSOR_DATA_TYPE dataType,
        py::bytes value) {
            return dml::FillValueConstant(graph.graph, std::move(sizes), dataType, ToScalarUnion(value));
        },
        "A tensor of one repeated value, computed rather than uploaded.",
        py::arg("graph"),
        py::kw_only(),
        py::arg("sizes"),
        py::arg("dtype"),
        py::arg("value"));

    module.def("fill_value_sequence", [](
        pydml::Graph& graph,
        dml::TensorDimensions sizes,
        DML_TENSOR_DATA_TYPE dataType,
        py::bytes valueStart,
        py::bytes valueDelta) {
            return dml::FillValueSequence(graph.graph, std::move(sizes), dataType,
                ToScalarUnion(valueStart), ToScalarUnion(valueDelta));
        },
        "An arithmetic sequence in memory order, computed rather than uploaded.",
        py::arg("graph"),
        py::kw_only(),
        py::arg("sizes"),
        py::arg("dtype"),
        py::arg("value_start"),
        py::arg("value_delta"));

    // Reductions and the operators with an output the shape of a question.
    //
    module.def("reduce", [](
        dml::Expression input,
        DML_REDUCE_FUNCTION function,
        std::vector<uint32_t> axes,
        DML_TENSOR_DATA_TYPE outputDataType) {
            return dml::Reduce(input, function, axes, outputDataType);
        },
        "Reduce the given axes to an extent of 1 with the named function.",
        py::arg("input"),
        py::kw_only(),
        py::arg("function"),
        py::arg("axes") = std::vector<uint32_t>{},
        py::arg("output_dtype") = DML_TENSOR_DATA_TYPE_UNKNOWN);

    module.def("cumulative_summation", [](
        dml::Expression input,
        uint32_t axis,
        DML_AXIS_DIRECTION axisDirection,
        bool hasExclusiveSum) {
            return dml::CumulativeSummation(input, axis, axisDirection, hasExclusiveSum);
        },
        "The running sum along an axis.",
        py::arg("input"),
        py::kw_only(),
        py::arg("axis"),
        py::arg("axis_direction") = DML_AXIS_DIRECTION_INCREASING,
        py::arg("has_exclusive_sum") = false);

    module.def("cumulative_product", [](
        dml::Expression input,
        uint32_t axis,
        DML_AXIS_DIRECTION axisDirection,
        bool hasExclusiveProduct) {
            return dml::CumulativeProduct(input, axis, axisDirection, hasExclusiveProduct);
        },
        "The running product along an axis.",
        py::arg("input"),
        py::kw_only(),
        py::arg("axis"),
        py::arg("axis_direction") = DML_AXIS_DIRECTION_INCREASING,
        py::arg("has_exclusive_product") = false);

    module.def("non_zero_coordinates", [](dml::Expression input) {
            auto outputs = dml::NonZeroCoordinates(input);
            return py::make_tuple(outputs.count, outputs.coordinates);
        },
        "How many elements are non-zero and where they are; the wrapper layer shapes the outputs.",
        py::arg("input"));

    module.def("quantize_linear", [](
        dml::Expression input,
        dml::Expression scale,
        dml::Expression zeroPoint,
        DML_TENSOR_DATA_TYPE outputDataType) {
            return dml::QuantizeLinear(input, scale, zeroPoint, outputDataType);
        },
        "Quantize the input as round(input / scale) + zero_point.",
        py::arg("input"),
        py::arg("scale"),
        py::arg("zero_point"),
        py::kw_only(),
        py::arg("output_dtype") = DML_TENSOR_DATA_TYPE_UINT8);

    module.def("dequantize_linear", &dml::DequantizeLinear,
        "Dequantize the input as (input - zero_point) * scale.",
        py::arg("input"),
        py::arg("scale"),
        py::arg("zero_point"));

    module.def("dequantize", [](
        dml::Expression input,
        std::vector<dml::Expression> quantizationParameters,
        DML_QUANTIZATION_TYPE quantizationType) {
            return dml::Dequantize(input, quantizationParameters, quantizationType);
        },
        "Dequantize the input by a list of parameters the quantization type decides the length of; the wrapper layer documents them.",
        py::arg("input"),
        py::arg("quantization_parameters"),
        py::kw_only(),
        py::arg("quantization_type"));

    module.def("random_generator", [](
        dml::Expression input_state,
        dml::TensorDimensions outputSizes,
        bool outputState,
        DML_RANDOM_GENERATOR_TYPE type) {
            auto outputs = dml::RandomGenerator(input_state, std::move(outputSizes), outputState, type);
            // As with max_pooling: an unrequested output is None.
            return py::make_tuple(
                outputs.values,
                outputState ? py::cast(outputs.state) : py::none());
        },
        "Uniform random uint32s from a generator state tensor; the wrapper layer shapes the outputs.",
        py::arg("input_state"),
        py::kw_only(),
        py::arg("output_sizes"),
        py::arg("output_state") = true,
        py::arg("type") = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10);

    module.def("roi_align", [](
        dml::Expression input,
        dml::Expression roi,
        dml::Expression batchIndices,
        DML_REDUCE_FUNCTION reductionFunction,
        DML_INTERPOLATION_MODE interpolationMode,
        float spatialScaleX,
        float spatialScaleY,
        float inputPixelOffset,
        float outputPixelOffset,
        float outOfBoundsInputValue,
        uint32_t minimumSamplesPerOutput,
        uint32_t maximumSamplesPerOutput,
        bool alignRegionsToCorners,
        uint32_t outputHeight,
        uint32_t outputWidth) {
            return dml::RoiAlign(input, roi, batchIndices, reductionFunction, interpolationMode,
                spatialScaleX, spatialScaleY, inputPixelOffset, outputPixelOffset,
                outOfBoundsInputValue, minimumSamplesPerOutput, maximumSamplesPerOutput,
                alignRegionsToCorners, outputHeight, outputWidth);
        },
        "Pool each region of interest to one output_height x output_width tile; the wrapper layer documents the parameters.",
        py::arg("input"),
        py::arg("roi"),
        py::arg("batch_indices"),
        py::kw_only(),
        py::arg("reduction_function"),
        py::arg("interpolation_mode"),
        py::arg("spatial_scale_x"),
        py::arg("spatial_scale_y"),
        py::arg("input_pixel_offset"),
        py::arg("output_pixel_offset"),
        py::arg("out_of_bounds_input_value"),
        py::arg("minimum_samples_per_output"),
        py::arg("maximum_samples_per_output"),
        py::arg("align_regions_to_corners"),
        py::arg("output_height"),
        py::arg("output_width"));

    // Quantized convolution. The zero points are optional because a symmetric
    // scheme has none; the scales never are.
    //
    module.def("convolution_integer", [](
        dml::Expression input,
        dml::Expression filter,
        dml::Optional<dml::Expression> inputZeroPoint,
        dml::Optional<dml::Expression> filterZeroPoint,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> dilations,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        uint32_t groupCount,
        dml::TensorDimensions outputSizes) {
            return dml::ConvolutionInteger(input, inputZeroPoint, filter, filterZeroPoint,
                strides, dilations, startPadding, endPadding, groupCount, std::move(outputSizes));
        },
        "Convolve an integer input with an integer filter into int32 sums; the wrapper layer documents the parameters.",
        py::arg("input"),
        py::arg("filter"),
        py::arg("input_zero_point") = dml::NullOpt,
        py::arg("filter_zero_point") = dml::NullOpt,
        py::kw_only(),
        py::arg("strides") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("group_count") = 1,
        py::arg("output_sizes") = dml::TensorDimensions{});

    module.def("quantized_linear_convolution", [](
        dml::Expression input,
        dml::Expression inputScale,
        dml::Expression filter,
        dml::Expression filterScale,
        dml::Expression outputScale,
        dml::Optional<dml::Expression> inputZeroPoint,
        dml::Optional<dml::Expression> filterZeroPoint,
        dml::Optional<dml::Expression> bias,
        dml::Optional<dml::Expression> outputZeroPoint,
        DML_TENSOR_DATA_TYPE outputDataType,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> dilations,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        uint32_t groupCount,
        dml::TensorDimensions outputSizes) {
            return dml::QuantizedLinearConvolution(input, inputScale, inputZeroPoint,
                filter, filterScale, filterZeroPoint, bias, outputScale, outputZeroPoint,
                outputDataType, strides, dilations, startPadding, endPadding, groupCount,
                std::move(outputSizes));
        },
        "Convolve in the quantized domain and requantize the result; the wrapper layer documents the parameters.",
        py::arg("input"),
        py::arg("input_scale"),
        py::arg("filter"),
        py::arg("filter_scale"),
        py::arg("output_scale"),
        py::arg("input_zero_point") = dml::NullOpt,
        py::arg("filter_zero_point") = dml::NullOpt,
        py::arg("bias") = dml::NullOpt,
        py::arg("output_zero_point") = dml::NullOpt,
        py::kw_only(),
        py::arg("output_dtype"),
        py::arg("strides") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("group_count") = 1,
        py::arg("output_sizes") = dml::TensorDimensions{});

    // The backward passes DirectML implements. Nothing here differentiates a
    // graph: each of these is the gradient of one forward operator, and the
    // chain rule stays with whoever is building the training step.
    //
    module.def("clip_grad", &dml::ClipGrad,
        "The gradient of clip: input_gradient where the input was inside [min, max], zero elsewhere.",
        py::arg("input"),
        py::arg("input_gradient"),
        py::kw_only(),
        py::arg("min"),
        py::arg("max"));

    module.def("batch_normalization_grad", [](
        dml::Expression input,
        dml::Expression inputGradient,
        dml::Expression mean,
        dml::Expression variance,
        dml::Expression scale,
        float epsilon) {
            auto outputs = dml::BatchNormalizationGrad(input, inputGradient, mean, variance, scale, epsilon);
            return py::make_tuple(outputs.gradient, outputs.scaleGradient, outputs.biasGradient);
        },
        "The gradient of batch_normalization; the wrapper layer shapes the outputs.",
        py::arg("input"),
        py::arg("input_gradient"),
        py::arg("mean"),
        py::arg("variance"),
        py::arg("scale"),
        py::kw_only(),
        py::arg("epsilon") = 1e-5f);

    module.def("batch_normalization_training", [](
        dml::Expression input,
        dml::Expression scale,
        dml::Expression bias,
        dml::Optional<dml::Expression> fusedAdd,
        float epsilon,
        std::optional<dml::FusedActivation> fusedActivation) {
            auto outputs = dml::BatchNormalizationTraining(input, scale, bias, fusedAdd, epsilon,
                fusedActivation.value_or(dml::FusedActivation::None()));
            return py::make_tuple(outputs.output, outputs.mean, outputs.variance);
        },
        "Batch normalization over statistics taken from the batch itself; the wrapper layer shapes the outputs.",
        py::arg("input"),
        py::arg("scale"),
        py::arg("bias"),
        py::arg("fused_add") = dml::NullOpt,
        py::kw_only(),
        py::arg("epsilon") = 1e-5f,
        py::arg("fused_activation") = py::none());

    module.def("batch_normalization_training_grad", [](
        dml::Expression input,
        dml::Expression inputGradient,
        dml::Expression mean,
        dml::Expression variance,
        dml::Expression scale,
        float epsilon) {
            auto outputs = dml::BatchNormalizationTrainingGrad(input, inputGradient, mean, variance, scale, epsilon);
            return py::make_tuple(outputs.gradient, outputs.scaleGradient, outputs.biasGradient);
        },
        "The gradient of batch_normalization_training; the wrapper layer shapes the outputs.",
        py::arg("input"),
        py::arg("input_gradient"),
        py::arg("mean"),
        py::arg("variance"),
        py::arg("scale"),
        py::kw_only(),
        py::arg("epsilon") = 1e-5f);

    module.def("resample_grad", [](
        dml::Expression input_gradient,
        dml::TensorDimensions outputSizes,
        DML_INTERPOLATION_MODE mode,
        std::vector<float> scales,
        std::vector<float> inputPixelOffsets,
        std::vector<float> outputPixelOffsets) {
            return dml::ResampleGrad(input_gradient, std::move(outputSizes), mode,
                scales, inputPixelOffsets, outputPixelOffsets);
        },
        "The gradient of resample: every output element's gradient summed back onto the inputs it read; the wrapper layer documents the offsets.",
        py::arg("input_gradient"),
        py::kw_only(),
        py::arg("output_sizes"),
        py::arg("mode"),
        py::arg("scales") = std::vector<float>{},
        py::arg("input_pixel_offsets") = std::vector<float>{},
        py::arg("output_pixel_offsets") = std::vector<float>{});

    module.def("slice_grad", [](
        dml::Expression input_gradient,
        dml::TensorDimensions outputGradientSizes,
        std::vector<uint32_t> inputWindowOffsets,
        std::vector<uint32_t> inputWindowSizes,
        std::vector<int32_t> inputWindowStrides) {
            return dml::SliceGrad(input_gradient, std::move(outputGradientSizes),
                inputWindowOffsets, inputWindowSizes, inputWindowStrides);
        },
        "The gradient of slice: the window's gradient scattered back into a tensor of zeros.",
        py::arg("input_gradient"),
        py::kw_only(),
        py::arg("output_gradient_sizes"),
        py::arg("input_window_offsets"),
        py::arg("input_window_sizes"),
        py::arg("input_window_strides"));

    module.def("roi_align_grad", [](
        dml::Expression inputGradient,
        dml::Expression roi,
        dml::Expression batchIndices,
        dml::Optional<dml::Expression> input,
        DML_REDUCE_FUNCTION reductionFunction,
        DML_INTERPOLATION_MODE interpolationMode,
        float spatialScaleX,
        float spatialScaleY,
        float inputPixelOffset,
        float outputPixelOffset,
        uint32_t minimumSamplesPerOutput,
        uint32_t maximumSamplesPerOutput,
        bool alignRegionsToCorners,
        uint32_t batchSize,
        uint32_t imageHeight,
        uint32_t imageWidth,
        bool computeOutputGradient,
        bool computeOutputRoiGradient) {
            auto outputs = dml::RoiAlignGrad(input, inputGradient, roi, batchIndices,
                reductionFunction, interpolationMode, spatialScaleX, spatialScaleY,
                inputPixelOffset, outputPixelOffset, minimumSamplesPerOutput,
                maximumSamplesPerOutput, alignRegionsToCorners, batchSize, imageHeight,
                imageWidth, computeOutputGradient, computeOutputRoiGradient);
            // As with max_pooling: an unrequested output is None.
            return py::make_tuple(
                computeOutputGradient ? py::cast(outputs.outputGradient) : py::none(),
                computeOutputRoiGradient ? py::cast(outputs.outputROIGradient) : py::none());
        },
        "The gradient of roi_align, towards the feature map and the boxes; the wrapper layer documents the parameters.",
        py::arg("input_gradient"),
        py::arg("roi"),
        py::arg("batch_indices"),
        py::arg("input") = dml::NullOpt,
        py::kw_only(),
        py::arg("reduction_function"),
        py::arg("interpolation_mode"),
        py::arg("spatial_scale_x"),
        py::arg("spatial_scale_y"),
        py::arg("input_pixel_offset"),
        py::arg("output_pixel_offset"),
        py::arg("minimum_samples_per_output"),
        py::arg("maximum_samples_per_output"),
        py::arg("align_regions_to_corners"),
        py::arg("batch_size"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("compute_output_gradient") = true,
        py::arg("compute_output_roi_gradient") = false);
}
