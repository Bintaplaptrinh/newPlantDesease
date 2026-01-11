## NewPlantDesease

### Cấu trúc thư mục

- `data/`: dữ liệu + các artifacts cho dashboard (`data/web/`). Dataset lớn (Kaggle) không commit.
- `src/`: mã nguồn Python (EDA, preprocessing, feature engineering, evaluation, inference server).
- `web/`: dashboard React (Vite).

### Cài đặt (Python)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Chạy backend (API)

```bash
python -m src.main serve --host 127.0.0.1 --port 5000
```

- API: `http://localhost:5000`

### Chạy dashboard (Web)

```bash
cd web/NewPlantDesease
pnpm install
pnpm dev
```

- Web (Vite dev): `http://localhost:5173`

### Dataset

- Nếu dataset quá lớn, không commit vào Git. Bạn có thể để ngoài repo hoặc dưới `data/kaggle/` (đã bị ignore).
- Xem `data/DATASET_PATHS.txt` để cấu hình đường dẫn dữ liệu trong máy.

## Docker (dev)

Chạy cả backend + frontend dev server:

```bash
docker compose up --build
```

- Backend: http://localhost:5000
- Web (Vite dev): http://localhost:5173

Ghi chú: trong container phải bind `0.0.0.0` (compose đã cấu hình sẵn), nên khác với lệnh local dùng `127.0.0.1`.