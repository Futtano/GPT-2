import pytest
import torch
from gpt_2.sattn import (
    SelfAttention_v1, 
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention)

@pytest.fixture
def attn_args():
    return {
        'd_in': 3,
        'd_out': 3,
        'context_length': 6,
        'dropout': 0.0,
        'num_heads': 3,
    }

@pytest.mark.parametrize(
    'input',
    [
        torch.ones((4, 3)),
        torch.ones((6, 3)),
    ]
)

def test_sattnv1(input, attn_args):
    sattn_v1 = SelfAttention_v1(attn_args['d_in'], attn_args['d_out'])
    context_vecs = sattn_v1(input)
    # context vector embedding dimension matches the provided d_out
    assert context_vecs.shape[-1] == attn_args['d_out']

    # context vector length matches inputs token_nums
    assert context_vecs.shape[1] == input.shape[1]
    
    # there are not NaN or torch.inf
    assert (~torch.isnan(context_vecs)).all()
    assert (~torch.isinf(context_vecs)).all()


@pytest.mark.parametrize(
    'input',
    [
        torch.ones((4, 3)),
        torch.ones((6, 3)),
    ]
)

def test_sattnv2(input, attn_args):
    sattn_v2 = SelfAttention_v2(attn_args['d_in'], attn_args['d_out'])
    context_vecs = sattn_v2(input)
    # context vector embedding dimension matches the provided d_out
    assert context_vecs.shape[-1] == attn_args['d_out']

    # context vector length matches inputs token_nums
    assert context_vecs.shape[1] == input.shape[1]
    
    # there are not NaN or torch.inf
    assert (~torch.isnan(context_vecs)).all()
    assert (~torch.isinf(context_vecs)).all()


@pytest.mark.parametrize(
    'input',
    [
        torch.ones((3, 4, 3)),
        torch.ones((4, 6, 3)),
    ]
)

def test_causalattn(input, attn_args):
    causal_attn = CausalAttention(
        d_in = attn_args['d_in'],
        d_out = attn_args['d_out'],
        context_length=attn_args['context_length'],
        dropout=attn_args['dropout']
    )
    context_vecs = causal_attn(input)
    # context vector embedding dimension matches the provided d_out
    assert context_vecs.shape[-1] == attn_args['d_out']

    # context vector length matches inputs token_nums
    assert context_vecs.shape[1] == input.shape[1]

    # the number of output batches is correct
    assert context_vecs.shape[0] == input.shape[0]
    
    # there are not NaN or torch.inf
    assert (~torch.isnan(context_vecs)).all()
    assert (~torch.isinf(context_vecs)).all()


@pytest.mark.parametrize(
    'input',
    [
        torch.ones((3, 4, 3)),
        torch.ones((4, 6, 3)),
    ]
)

def test_mhawrapper(input, attn_args):
    mha = MultiHeadAttentionWrapper(
        **attn_args,
    )
    context_vecs = mha(input)
    # context vector embedding dimension matches the provided d_out*num_heads
    assert context_vecs.shape[-1] == attn_args['d_out'] * attn_args['num_heads']

    # context vector length matches inputs token_nums
    assert context_vecs.shape[1] == input.shape[1]

    # the number of output batches is correct
    assert context_vecs.shape[0] == input.shape[0]
    
    # there are not NaN or torch.inf
    assert (~torch.isnan(context_vecs)).all()
    assert (~torch.isinf(context_vecs)).all()


@pytest.fixture
def mhattn_args():
    return {
        'd_in': 3,
        'context_length': 6,
        'dropout': 0.0,
        'num_heads': 3,
    }

@pytest.mark.parametrize(
    'input, d_out',
    [
        (torch.ones((3, 4, 3)), 6),
        pytest.param(
            torch.ones((4, 6, 3)),
            5,
            marks=pytest.mark.xfail(
                reason="d_out % num_heads != 0 should raise an assertion error"
            )
        ),

    ]
)

def test_mha(input, d_out, mhattn_args):
    mha = MultiHeadAttention(
        **mhattn_args, d_out=d_out
    )
    context_vecs = mha(input)

    # context vector embedding dimension matches the provided d_out
    assert context_vecs.shape[-1] == d_out

    # context vector length matches inputs token_nums
    assert context_vecs.shape[1] == input.shape[1]

    # the number of output batches is correct
    assert context_vecs.shape[0] == input.shape[0]
    
    # there are not NaN or torch.inf
    assert (~torch.isnan(context_vecs)).all()
    assert (~torch.isinf(context_vecs)).all()