# S13 — Bảo vệ MovieGraph

## Slide deck

Deck 12 slide đã lưu trong Spaces: `MovieGraph - Bảo vệ đồ án Big Data` (`page_01af54e84f488191b0761ff7e2e66b38`). Hai biểu đồ REC-01 và EXP-01 là chart gốc có thể sửa. Mỗi slide đã được render và rà trực quan ở 1280×720. Trình tự: đề tài → RQ → dữ liệu → pipeline → temporal split → train → REC-01 → EXP-01 → EXP-02 → graph feedback → giới hạn → demo/kết luận.

[Bản PDF 12 trang](MovieGraph-defense.pdf) và 12 PNG trong `docs/defense-slides/` được đóng gói để trình chiếu ngoại tuyến. Bản có thể chỉnh sửa là deck Spaces; lần export PPTX qua artifact session chưa tạo được tệp cục bộ, nên package không nhận PDF là PPTX.

## Kịch bản demo 4–5 phút

1. Mở Streamlit bằng lệnh trong `README.md`; chỉ vào REC-01 checkpoint step 300 và graph version 81. Chọn warm user, xem Top-20 và movie metadata.
2. Giải thích 773 candidates là train catalog, train-seen items đã mask, title/genres chỉ bổ sung sau ranking.
3. Ghi một batch rating hợp lệ trên graph demo branch; xem canonical graph version 81→82, model graph version vẫn 81. Không nói Top-20 đã học từ batch mới.
4. Replay cùng request để trình bày `NO_OP` nếu demo route cho phép; nếu không, dùng báo cáo S11 đã seal thay cho thao tác live.
5. Kết lại bằng hai kết quả cần nhớ: LightGCN test thấp hơn Popularity chút ít; Incremental chỉ thắng ở điểm H=2M, Δ=1k của run riêng.

## Câu hỏi dự kiến

**Vì sao đề tài gọi là Big Data khi model chỉ train 1M?** MovieLens 32M là nguồn và D02 đã profile/kiểm toàn bộ 32.000.204 ratings. Huấn luyện REC-01 chỉ trên canonical prefix 1M do giới hạn tài nguyên, và benchmark graph tới 2M. Không tuyên bố đã train toàn bộ 32M.

**Vì sao LightGCN lại thua Popularity?** Trong test warm cohort 403 users và 773 train candidates, Popularity hơn 0,000099 Recall@20 và 0,001736 NDCG@20. Một seed, 500 steps tối đa, warm catalog nhỏ, popularity mạnh; chưa có kiểm định hay nhiều seeds. Kết quả được giữ nguyên theo protocol khóa trước run, không điều chỉnh sau khi thấy test.

**Có rò rỉ test khi dựng graph/Popularity không?** Cutoff 80/90% distinct timestamps được xác định trước. Train graph, candidates và Popularity chỉ từ train. Validation chọn checkpoint step 300; test chỉ tính sau khi reload checkpoint đã chọn. Tests kiểm no-future-leakage; output seal xác nhận nguồn và config.

**Tại sao chọn ngưỡng rating ≥4,0?** Đây là quyết định DEC-005 để quy đổi explicit rating thành binary positive edge; chưa chứng minh ngưỡng tối ưu. Rating thấp vẫn được nhận làm feedback event nhưng không tạo positive edge.

**Incremental có luôn nhanh hơn Full không?** Không. Trong core grid H≤1M, Full nhanh hơn mọi cặp đo. Run H=2M, Δ=1k cho Incremental thắng 5/5; Δ=10k gần hòa. Prebuild history của Incremental nằm ngoài timer, còn export toàn bộ state nằm trong cả hai methods. Cache OS không reset được, nên chỉ mô tả phạm vi đo.

**Làm sao chứng minh cập nhật đúng?** Full Rebuild độc lập từ cùng prefix được so exact registry, edges và degrees. D05 kiểm 35 boundaries và 70 replay. S11 với 802.988 train events + 20 feedback events khớp từng dòng và replay trả NO_OP. 60 local tests PASS gồm crash/tamper paths.

**Graph version 82 có nghĩa model đã cập nhật chưa?** Chưa. Batch feedback cập nhật canonical graph branch từ v81 lên v82; normalized model graph và checkpoint vẫn v81, step 300. Top-20 được giữ nguyên. Đây là chủ ý để app không gán nhãn sai cho một model chưa refresh.

**R=5 trong EXP-02 đổi chất lượng gợi ý bao nhiêu?** Thí nghiệm đó chỉ đo normalization/export: 1,648 s so 5,283 s và staleness tối đa bốn batch/40k events; normalized output ở v5/v10 khớp. Chưa đo thay đổi Recall/NDCG sau refresh nên không đưa claim về chất lượng.

**Tại sao không dùng raw 32M trong Kaggle Dataset?** Bundle S03 chỉ chứa canonical 1M prefix, mã và manifest đã kiểm hash. Quy mô/nguồn được mô tả tách biệt; raw 32M không cần upload để chạy REC-01.

**Có thể tái lập kết quả không?** Protocol, source hash, config, checkpoint/output manifest, lệnh và raw benchmark CSV được ghi trong README và các tài liệu S01–S11. S14 cần đóng gói artifacts đúng manifest; không cần tải lại raw 32M nếu chỉ kiểm REC-01 output, nhưng dựng từ đầu cần nguồn GroupLens theo điều khoản của họ.

**Đóng góp kỹ thuật chính là gì?** Canonical temporal events, versioned incremental graph có ledger và đối chứng exact Full Rebuild; LightGCN đã train và test theo protocol; ứng dụng thể hiện rõ ranh giới graph/model freshness; benchmark báo cả trường hợp Incremental không thắng.

## Điều cần điền trước buổi bảo vệ

Họ tên/MSSV, lớp, giảng viên, nhóm và vai trò thực tế. Không dùng dữ liệu ví dụ trong mẫu báo cáo.
