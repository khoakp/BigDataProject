# D01 — Môi trường và kiểm tra khả thi ban đầu

**Cập nhật D07 ngày 20/11/2026:** venv dự án hiện có PyTorch 2.8.0+cu126 với CUDA 12.6. `torch.cuda.is_available()` trả `True`, và phép nhân tensor chạy thành công trên RTX 4050 Laptop GPU với driver 566.07. PyTorch cấp hệ thống vẫn là bản CPU được quan sát ở D01. Kết quả model smoke và cấu hình mới ở [D07](d07-lightgcn-smoke.md). Bảng bên dưới giữ bằng chứng lịch sử D01.

Ngày kiểm tra: 19/11/2026 (Asia/Saigon). D01 bắt đầu ngày này; đây là kiểm tra môi trường, không phải benchmark hay GPU model smoke D07.

| Thành phần | Quan sát | Cách kiểm tra / giới hạn |
|---|---|---|
| OS | Windows, build 26300, 64-bit | `platform.platform()` trả `Windows-10-10.0.26300-SP0`; tên Windows 10 do API Python, không suy ra edition |
| CPU | Intel64 Family 6 Model 183 Stepping 1; 14 physical / 20 logical cores | `platform.processor()`, `psutil.cpu_count()`; tên thương mại chưa lấy được |
| RAM | 16,780,570,624 bytes (~15.63 GiB) | `psutil.virtual_memory().total` |
| D: | 161,060,220,928 bytes tổng; 91,772,035,072 bytes trống lúc kiểm tra | `psutil.disk_usage('D:/')`; giá trị trống thay đổi theo thời gian |
| GPU/driver | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB theo `nvidia-smi`; driver 566.07, driver CUDA compatibility 12.7 | `nvidia-smi`; đây không phải CUDA runtime của PyTorch |
| Python | 3.10.11 (system và `../venv`) | `python --version`, `../venv/Scripts/python.exe --version` |
| DuckDB | 1.5.6 trong `../venv` | import thành công |
| PyTorch system | 2.8.0+cpu; `torch.version.cuda = None`, `torch.cuda.is_available() = False` | CUDA smoke: unavailable vì bản CPU |
| PyTorch `../venv` | chưa cài | import thất bại `ModuleNotFoundError` |
| PyG `../venv` / system | chưa cài | import thất bại `ModuleNotFoundError` |

`Get-CimInstance` và `Get-Volume` không truy cập được CIM resource trong môi trường lệnh hiện tại; CPU/RAM/disk được lấy bằng `psutil`. Không coi tên GPU trong `nvidia-smi` là bằng chứng PyTorch CUDA đã hoạt động.

## Torch smoke log

Lệnh được chạy với Python hệ thống, dùng tensor CPU `4×4`, `x = arange(1,17).reshape(4,4).requires_grad_()`, `y = x.square().mean(); y.backward()`. Kết quả: **PASS**, loss `93.5`, gradient shape `4×4`, phần tử gradient đầu `0.125`. Lần thử trước dùng tensor không phải leaf nên `x.grad` không được lưu; đã sửa cách tạo input và chạy lại đạt. CUDA smoke: **CUDA_UNAVAILABLE** do `torch 2.8.0+cpu`; chưa thử forward/backward trên GPU hay LightGCN.

## Tái lập kiểm tra

```powershell
python --version
..\venv\Scripts\python.exe --version
nvidia-smi
@'
import platform, psutil, torch
print(platform.platform(), platform.processor())
print(psutil.cpu_count(logical=False), psutil.cpu_count(), psutil.virtual_memory().total)
print(psutil.disk_usage('D:/'))
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
x = torch.arange(1,17,dtype=torch.float32).reshape(4,4).requires_grad_()
y = x.square().mean(); y.backward()
assert x.grad is not None and tuple(x.grad.shape) == (4,4)
print(float(y), float(x.grad.flatten()[0]))
'@ | python -
```

**Việc chuyển sang D07:** cài PyTorch CUDA, chạy LightGCN sparse smoke và ghi VRAM thực. D07 đã dùng phương án PyTorch native theo [R-04](06-decisions-and-risks.md).
