"""
Export a trained rain classifier checkpoint (.pth) to ONNX format.
"""

import argparse
from pathlib import Path

import torch
import timm


MODEL_NAME = 'mobilenetv4_conv_medium.e500_r224_in1k'
INPUT_SIZE = 224
NUM_CLASSES = 2


def create_model():
    """Create model architecture used during training."""
    return timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )


def export_checkpoint_to_onnx(
    checkpoint_path,
    output_path,
    opset_version=18,
    dynamic_batch=True,
    device='cpu',
):
    """Load checkpoint and export to ONNX."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('onnx is required. Install with: pip install onnx onnxruntime onnxscript') from exc

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    requested_device = torch.device(device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but unavailable, falling back to CPU for checkpoint load.')
        requested_device = torch.device('cpu')

    print(f'Loading checkpoint: {checkpoint_path}')
    print(f'Checkpoint load device: {requested_device}')

    model = create_model()
    state_dict = torch.load(checkpoint_path, map_location=requested_device)
    model.load_state_dict(state_dict)
    model.eval()

    # Exporting from CPU avoids device-specific export inconsistencies.
    model = model.to('cpu')
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Exporting ONNX model to: {output_path}')
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
    )

    print('Validating ONNX graph...')
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    exported_opset = onnx_model.opset_import[0].version if onnx_model.opset_import else opset_version

    print('ONNX export complete.')
    print(f'File: {output_path}')
    print(f'Opset: {exported_opset}')
    print(f'Dynamic batch axis: {dynamic_batch}')


def main():
    parser = argparse.ArgumentParser(description='Export trained rain classifier checkpoint to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_rain_classifier.pth',
        help='Path to input checkpoint (.pth)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='best_rain_classifier.onnx',
        help='Path to output ONNX model',
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=18,
        help='ONNX opset version',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device used to load checkpoint (cuda or cpu)',
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--dynamic-batch',
        dest='dynamic_batch',
        action='store_true',
        help='Enable dynamic batch axis (default)',
    )
    group.add_argument(
        '--fixed-batch',
        dest='dynamic_batch',
        action='store_false',
        help='Export with fixed batch size of 1',
    )
    parser.set_defaults(dynamic_batch=True)

    args = parser.parse_args()

    export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
        device=args.device,
    )


if __name__ == '__main__':
    main()
