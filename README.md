python -m src.main serve --host 127.0.0.1 --port 5000

## Docker (dev)

Chạy cả backend + frontend dev server:

```bash
docker compose up --build
```

- Backend: http://localhost:5000
- Web (Vite dev): http://localhost:5173

Ghi chú: trong container phải bind `0.0.0.0` (compose đã cấu hình sẵn), nên khác với lệnh local dùng `127.0.0.1`.