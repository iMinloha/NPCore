"""
Export a complex PyTorch model to ONNX for NPCore import testing.

Model: SmallCNN
  Conv2d(1→8,k=3,p=1) → BN → ReLU → MaxPool
  Conv2d(8→16,k=3,p=1) → BN → ReLU → MaxPool
  Flatten → Linear(16*7*7→128) → ReLU → Linear(128→10) → Softmax

Outputs:
  - torch_cnn.onnx          ONNX model
  - torch_cnn_input.bin     test input tensor (float32 raw)
  - torch_cnn_output.bin    expected output tensor (float32 raw)
  - torch_cnn_shapes.txt    shape info for C++ loader
"""
import torch
import torch.nn as nn
import numpy as np
import os

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    model = SmallCNN()
    model.eval()

    # Dummy input: (1, 1, 28, 28) — batch=1, grayscale, 28x28
    dummy = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy)

    # Export ONNX
    onnx_path = os.path.join(out_dir, "torch_cnn.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"[OK] ONNX exported: {onnx_path}")

    # Save input tensor (raw float32)
    inp = dummy.numpy().astype(np.float32)
    inp_path = os.path.join(out_dir, "torch_cnn_input.bin")
    inp.tofile(inp_path)
    print(f"[OK] Input saved: {inp_path}  shape={list(inp.shape)}")

    # Save output tensor (raw float32)
    out = output.numpy().astype(np.float32)
    out_path = os.path.join(out_dir, "torch_cnn_output.bin")
    out.tofile(out_path)
    print(f"[OK] Output saved: {out_path}  shape={list(out.shape)}")

    # Save shapes info
    shape_path = os.path.join(out_dir, "torch_cnn_shapes.txt")
    with open(shape_path, 'w') as f:
        f.write(f"input {' '.join(str(s) for s in inp.shape)}\n")
        f.write(f"output {' '.join(str(s) for s in out.shape)}\n")
    print(f"[OK] Shapes saved: {shape_path}")

    # Verify ONNX
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f"[OK] ONNX checker passed")
    print(f"  nodes: {len(model_onnx.graph.node)}")
    for n in model_onnx.graph.node:
        print(f"    {n.op_type}: {list(n.input)} -> {list(n.output)}")
    print(f"  initializers: {len(model_onnx.graph.initializer)}")

if __name__ == "__main__":
    main()
