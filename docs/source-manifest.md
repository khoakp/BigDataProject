# D01 — MovieLens 32M source manifest

Kiểm tra ngày 19/11/2026. Dữ liệu nằm ngoài Git tại `../ml-32m/` tính từ root repo; repo không sao chép các CSV này. Nguồn tải ban đầu chưa có log URL/zip hash, nên manifest chỉ xác nhận hai file cục bộ khớp `checksums.txt` đi kèm.

| File | Bytes | MD5 tính bằng `Get-FileHash` | MD5 manifest | Trạng thái |
|---|---:|---|---|---|
| `ratings.csv` | 877076222 | `cf12b74f9ad4b94a011f079e26d4270a` | `cf12b74f9ad4b94a011f079e26d4270a` | khớp |
| `movies.csv` | 4242926 | `0df90835c19151f9d819d0822e190797` | `0df90835c19151f9d819d0822e190797` | khớp |

Header đọc trực tiếp: `ratings.csv`: `userId,movieId,rating,timestamp`; `movies.csv`: `movieId,title,genres`. README đi kèm công bố 32,000,204 ratings, 87,585 movies và 200,948 users; chưa đếm độc lập. `links.csv` và `tags.csv` không thuộc input D01 cần hash, chưa kiểm tra lại MD5.

Đã đọc `README.txt`, gồm phần Usage License và Citation. Điều kiện nêu trong file: dùng cho nghiên cứu, ghi nhận dataset khi công bố, không gợi ý GroupLens bảo trợ, và không dùng thương mại khi chưa có phép. Citation do README đề xuất: Harper, F. M. & Konstan, J. A. (2015), *The MovieLens Datasets: History and Context*, DOI `10.1145/2827872`. Cần đối chiếu yêu cầu trích dẫn của giảng viên khi nộp.

Lệnh lặp lại:

```powershell
Get-FileHash -Algorithm MD5 ..\ml-32m\ratings.csv, ..\ml-32m\movies.csv
Get-Content ..\ml-32m\checksums.txt
Get-Content ..\ml-32m\README.txt
```

Repo HEAD lúc kiểm tra: `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`; thay đổi D01 chưa commit. MD5 bảo vệ đối chiếu file với manifest cục bộ, không chứng minh dữ liệu đã được sort, sạch hoặc không trùng.
