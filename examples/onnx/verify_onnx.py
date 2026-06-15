"""NPCore ONNX Verification Script

Validates ONNX files exported by NPCore.
Run:  python verify_onnx.py
Requires: onnx (pip install onnx)
Optional: onnxruntime (pip install onnxruntime)
"""
import os, sys, struct

def check_magic(path):
    """Basic check: file starts with ONNX IR version varint."""
    with open(path, 'rb') as f:
        first_byte = f.read(1)[0]
    # Protobuf field 1, varint wire type 0 => (1<<3)|0 = 0x08
    if first_byte == 0x08:
        print(f"  [OK] {path} — valid protobuf header")
        return True
    print(f"  [FAIL] {path} — unexpected first byte: 0x{first_byte:02x}")
    return False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[WARN] onnx not installed. Install with: pip install onnx")
    print("[INFO] Running basic header checks only.\n")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Also check project root (one level up from examples/onnx)
    project_root = os.path.dirname(os.path.dirname(test_dir))
    files = ['test_mlp.onnx', 'test_cnn.onnx', 'test_bn.onnx']
    all_ok = True

    for fname in files:
        path = os.path.join(test_dir, fname)
        if not os.path.exists(path):
            path = os.path.join(project_root, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} — file not found (run test_onnx.exe first)")
            continue

        print(f"\n--- {fname} ---")
        if not check_magic(path):
            all_ok = False
            continue

        if HAS_ONNX:
            try:
                model = onnx.load(path)
                onnx.checker.check_model(model)
                print(f"  [OK] onnx.checker passed")
                print(f"  ir_version={model.ir_version}, producer={model.producer_name}")
                g = model.graph
                print(f"  graph: {g.name}, nodes={len(g.node)}, inputs={len(g.input)}, outputs={len(g.output)}")
                for n in g.node:
                    print(f"    {n.op_type}: {list(n.input)} -> {list(n.output)}")
            except Exception as e:
                print(f"  [FAIL] onnx.checker: {e}")
                all_ok = False

        if HAS_ORT:
            try:
                session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                print(f"  [OK] onnxruntime loaded successfully")
                inp = session.get_inputs()[0]
                print(f"  input: {inp.name} shape={inp.shape}")
                out = session.get_outputs()[0]
                print(f"  output: {out.name} shape={out.shape}")
            except Exception as e:
                print(f"  [WARN] onnxruntime: {e}")

    print(f"\n=== {'ALL OK' if all_ok else 'SOME FAILED'} ===")
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
