//-----------------------------------------------------------------------------
//
//  DML_OPERATOR_MULTIHEAD_ATTENTION, which DirectMLX does not wrap.
//
//-----------------------------------------------------------------------------

#pragma once

namespace pydml
{
    // The operator wants { 1, batch, tokens, width } and the graphs here carry
    // { batch, 1, tokens, width }. Those are the same bytes in the same order,
    // so moving the batch across is a view and not a copy -- but it does have to
    // happen, because both ends of a graph edge have to agree on the shape.
    // Nothing to do at batch 1, which is every shape but a guided one.
    inline dml::Expression MoveBatchAxis(dml::Expression x, bool toSecond)
    {
        dml::TensorDesc tensor = x.Impl()->GetOutputDesc();
        uint32_t batch = tensor.sizes[toSecond ? 0 : 1];
        if (batch == 1)
        {
            return x;
        }

        dml::TensorDimensions moved = tensor.sizes;
        moved[0] = toSecond ? 1 : batch;
        moved[1] = toSecond ? batch : 1;
        return dml::Reinterpret(x, std::move(moved), dml::NullOpt);
    }

    // Attention written out of gemm, softmax and gemm materializes the whole
    // score matrix, which grows with the square of the token count and is read
    // and written twice on its way through. DirectML has one operator for the
    // three; third_party/DirectMLX.h just has no helper that emits it, so this
    // is that helper, in the style of the ones next to it.
    //
    // Only unmasked attention is exposed. The descriptor also takes stacked QKV,
    // a bias, four kinds of mask, a relative position bias and a past key-value
    // cache, and every one of those is left null here.
    inline dml::Expression MultiHeadAttention(
        dml::Expression query,
        dml::Expression key,
        dml::Expression value,
        uint32_t headCount,
        float scale)
    {
        query = MoveBatchAxis(query, true);
        key = MoveBatchAxis(key, true);
        value = MoveBatchAxis(value, true);

        dml::detail::GraphBuilder* builder = query.Impl()->GetGraphBuilder();

        dml::TensorDesc queryTensor = query.Impl()->GetOutputDesc();
        dml::TensorDesc keyTensor = key.Impl()->GetOutputDesc();
        dml::TensorDesc valueTensor = value.Impl()->GetOutputDesc();

        // As many tokens as the query has, as wide as the value is.
        dml::TensorDimensions outputSizes = queryTensor.sizes;
        outputSizes.back() = valueTensor.sizes.back();

        dml::TensorDesc outputTensor(
            queryTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC desc = {};
        desc.QueryTensor = queryTensor.AsPtr<DML_TENSOR_DESC>();
        desc.KeyTensor = keyTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ValueTensor = valueTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Scale = scale;
        desc.HeadCount = headCount;
        desc.MaskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;

        // One entry per tensor in the descriptor, in the order they appear
        // there: the three inputs, the three stacked forms, bias, mask,
        // relative position bias, and the two halves of the past cache.
        dml::detail::NodeOutput* const inputs[] = {
            query.Impl(), key.Impl(), value.Impl(),
            nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            nullptr, nullptr,
        };

        dml::detail::NodeID node = builder->CreateOperatorNode(
            DML_OPERATOR_MULTIHEAD_ATTENTION, &desc, inputs);
        return MoveBatchAxis(
            builder->CreateNodeOutput(node, 0, std::move(outputTensor)), false);
    }
}
