# Yêu cầu cho Muse AI: cập nhật và chạy đối chứng reward vật lý

Repo: https://github.com/hongphuc-git/wavelab-rl-scm
Notebook: notebooks/WaveLab_RL_Colab.ipynb

Hãy lấy bản master mới nhất và cập nhật bản notebook trên Google Drive từ repo. Giữ nguyên kết quả, checkpoint và replay cũ. Không resume checkpoint tiến bộ tham số sang reward vật lý.

Notebook đã thêm EXPERIMENT: physics_blur, physics_nearest, physics_le2024 và parameter_progress. Ba nhánh physics phải dùng absolute + param_weight=0, r=-J-0.01. Baseline dùng 2*(e_trước-e_sau)-0.01. Không đưa nhãn vào observation hoặc dùng nhãn để chọn nghiệm khi đánh giá. Le2024 là bản cài đặt trên ảnh raster; chưa xác nhận tái lập nguyên bản bài báo.

Chạy từng nhánh từ đầu trên cùng 256 train/32 validation tách biệt, seed2026, epoch50 frozen, horizon8, action_scale0.15, 4 workers, batch8, 4000 bước; giữ các hyperparameter khác như notebook. Nếu thiếu GPU/dữ liệu/quyền truy cập hãy báo rõ. Chạy từng cell, KHÔNG Run all vì cell STOP cuối notebook sẽ dừng train.

Sau từng nhánh chạy cell đánh giá: xuất JSON, CSV, scatter của các checkpoint trên cùng validation và seed khởi tạo. Báo cáo initial, final, best-J riêng và best blur-NCC chung; so sánh MAE, RMSE, R2, MARE và mean normalized error. Không so trực tiếp trị số J khác định nghĩa. Nhãn chỉ dùng tính metric sau rollout.

Kiểm tra checkpoint/replay/config sao lưu Drive trước khi tắt runtime; không thay reward trong cùng thư mục. Hoàn tất pilot một seed rồi báo kết quả trước khi mở rộng. Sửa phần tài liệu khẳng định J Le2024 không dùng được: hiện chưa có bằng chứng train đối chứng để kết luận; 53% ranking trong tài liệu cũ không phải độ chính xác SAC. Không tự nhận đã chạy nếu mới sửa code.
