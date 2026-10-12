"""Independent architecture, mapping-boundary and gradient-flow calibration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.b0_model import (ResidualCNN, configure_reproducibility, data_complex_mse,
                          predict_channel, spatial_data_mask, training_step)
from src.ml_data import complex_to_tensor, tensor_to_complex
from src.pilots import build_fdm_pilot_layout


def setup_module():
    configure_reproducibility(10261001)


def test_torch_environment_and_float32_autograd():
    x = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
    (x*x).sum().backward()
    assert x.grad.item() == 4.0
    if torch.cuda.is_available():
        assert (x.detach().cuda()*2).cpu().item() == 4.0


def test_exact_architecture_and_independent_parameter_MAC_arithmetic():
    model = ResidualCNN()
    modules = list(model.layers)
    assert [type(m) for m in modules] == [nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU,
                                        nn.Conv2d, nn.ReLU, nn.Conv2d]
    convs = modules[::2]
    assert [(c.in_channels, c.out_channels) for c in convs] == [(8,32),(32,32),(32,32),(32,8)]
    assert all(c.kernel_size == (3,3) and c.stride == (1,1) and c.padding == (1,1)
               and c.dilation == (1,1) and c.groups == 1 and c.bias is not None for c in convs)
    expected_parameters = 2336 + 9248 + 9248 + 2312
    assert expected_parameters == 23144 == sum(p.numel() for p in model.parameters())
    independent_MAC = (8*32 + 32*32 + 32*32 + 32*8)*9*64*16
    actual_MAC = sum(c.weight.numel()*64*16 for c in convs)
    assert independent_MAC == 23592960 == actual_MAC


def test_four_density_models_do_not_share_parameters():
    models = [ResidualCNN() for _ in (2,4,8,16)]
    assert len({id(m.layers[0].weight) for m in models}) == 4
    with torch.no_grad():
        models[0].layers[-1].bias.fill_(1)
    assert all(torch.count_nonzero(m.layers[-1].bias).item() == 0 for m in models[1:])


@pytest.mark.parametrize("density", [2,4,8,16])
def test_zero_init_output_and_quantized_R3_bitwise_identity(density):
    h = np.random.default_rng(901).normal(size=(2,2,64,16)) + 1j*np.random.default_rng(902).normal(size=(2,2,64,16))
    base = complex_to_tensor(h)
    model = ResidualCNN()
    layout = build_fdm_pilot_layout(2,64,16,density,4)
    residual = model(base)
    prediction = predict_channel(model, base, layout.data_mask)
    assert residual.shape == base.shape and residual.dtype == torch.float32
    assert torch.count_nonzero(model.layers[-1].weight).item() == 0
    assert torch.count_nonzero(model.layers[-1].bias).item() == 0
    assert torch.count_nonzero(residual).item() == 0
    assert prediction.detach().numpy().tobytes() == base.numpy().tobytes()
    assert tensor_to_complex(prediction).tobytes() == tensor_to_complex(base).tobytes()


@pytest.mark.parametrize("density", [2,4,8,16])
def test_pilot_union_exactly_preserved_after_nonzero_learned_residual(density):
    model = ResidualCNN()
    with torch.no_grad():
        model.layers[-1].bias.fill_(2.0)
    base = torch.randn(2,8,64,16)
    layout = build_fdm_pilot_layout(2,64,16,density,4)
    result = predict_channel(model, base, layout.data_mask)
    union = torch.from_numpy(layout.pilot_union)
    assert torch.equal(result[..., union], base[..., union])
    assert torch.equal(result[..., ~union], base[..., ~union]+2)


def test_loss_is_complex_MSE_on_actual_data_mask_without_half_factor():
    layout = build_fdm_pilot_layout(2,64,16,16,4)
    prediction = torch.zeros(2,8,64,16)
    target = torch.empty_like(prediction)
    for link in range(4):
        target[:,2*link].fill_(link+1)
        target[:,2*link+1].fill_(2*(link+1))
    expected = (5 + 20 + 45 + 80)/4
    assert data_complex_mse(prediction,target,layout.data_mask).item() == expected
    target[..., torch.from_numpy(layout.pilot_union)] = 1e6
    assert data_complex_mse(prediction,target,layout.data_mask).item() == expected
    assert data_complex_mse(prediction,2*target,layout.data_mask).item() == 4*expected


def test_mask_rejects_noncommon_TX_mask_and_empty_mask():
    wrong = np.ones((2,64,16),dtype=bool); wrong[1,0,0] = False
    with pytest.raises(ValueError,match="common"):
        spatial_data_mask(wrong)
    with pytest.raises(ValueError):
        spatial_data_mask(np.zeros((64,16),dtype=bool))


def test_no_normalization_in_forward_and_nonfinite_or_wrong_input_rejected():
    model = ResidualCNN()
    observed = []
    hook = model.layers[0].register_forward_pre_hook(lambda m,args: observed.append(args[0]))
    base = 20*torch.randn(1,8,64,16)
    model(base)
    hook.remove()
    assert observed[0] is base
    for bad in (base.double(), base[:,:,1:], torch.full_like(base,float("nan"))):
        with pytest.raises(ValueError):
            model(bad)


def test_first_backward_final_gradient_then_second_backward_early_gradients():
    configure_reproducibility(10261001)
    model = ResidualCNN()
    base = torch.randn(2,8,64,16)
    target = 0.7*base + 0.2
    mask = build_fdm_pilot_layout(2,64,16,16,4).data_mask
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    data_complex_mse(predict_channel(model,base,mask),target,mask).backward()
    assert model.layers[-1].weight.grad.norm().item() > 0
    assert all(model.layers[i].weight.grad.norm().item() == 0 for i in (0,2,4))
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    data_complex_mse(predict_channel(model,base,mask),target,mask).backward()
    assert all(model.layers[i].weight.grad.norm().item() > 0 for i in (0,2,4,6))


def test_training_step_changes_parameters_and_returns_finite_loss():
    model = ResidualCNN()
    base = torch.randn(1,8,64,16)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss = training_step(model,optimizer,base,0.7*base,np.ones((64,16),dtype=bool))
    assert np.isfinite(loss) and torch.count_nonzero(model.layers[-1].weight).item() > 0


@pytest.fixture(scope="module")
def tiny_smoke():
    from scripts.validate_b0_pipeline import calibration_data,run_smoke
    from src.trajectories import prepare_trajectory
    base,truth,layout,records,sentinels = calibration_data(prepare_trajectory())
    first = run_smoke(base,truth,layout,torch.device("cpu"))
    repeated = run_smoke(base,truth,layout,torch.device("cpu"))
    return first,repeated,records,sentinels


def test_tiny_overfit_is_calibration_only_and_loss_decreases(tiny_smoke):
    first,_,records,sentinels = tiny_smoke
    assert len(records) == 8 and {r["namespace"] for r in records} == {"calibration"}
    assert {r["trajectory_id"] for r in records} == {0,1,2,3}
    assert first["all_checks_passed"] and first["final_loss"] <= 0.5*first["initial_loss"]
    assert all(r["machine_precision_match"] for r in sentinels)
    assert not first["performance_claim"]


def test_same_seed_smoke_trace_and_state_are_exactly_reproducible(tiny_smoke):
    a,b,_,_ = tiny_smoke
    assert a["loss_trace"] == b["loss_trace"]
    assert a["final_loss"] == b["final_loss"]
    assert a["final_model_state_sha256"] == b["final_model_state_sha256"]


def test_real_calibration_gradient_and_pilot_checks(tiny_smoke):
    a = tiny_smoke[0]
    assert a["gradient_checks"]["all_checks_passed"]
    assert a["pilot_preservation"]["all_union_pilot_values_exact_preserved"]
    assert all(a["zero_init_checks"].values())
