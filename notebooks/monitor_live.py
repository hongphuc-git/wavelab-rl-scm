# Tự cập nhật mỗi 3 giây. Ngắt cell này chỉ dừng hiển thị; train vẫn chạy nền.
import time, json
from IPython.display import clear_output, display, HTML
try:
    while True:
        clear_output(wait=True)
        progress_path=RUN/'progress.json'
        try: info=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        except (OSError,ValueError): info={}
        step=int(info.get('step',0)); pct=min(100,100*step/TOTAL_STEPS)
        display(HTML(f'<h3>EfficientNet V2-L epoch {EPOCH} + SAC</h3><progress value="{step}" max="{TOTAL_STEPS}" style="width:100%;height:24px"></progress><b>{step:,} / {TOTAL_STEPS:,} bước ({pct:.1f}%)</b>'))
        def number(x): return f'{x:.5f}' if isinstance(x,(int,float)) else 'chưa có'
        J=info.get('mean_J');err=info.get('mean_param_error')
        reward=-J-.01-2*err if isinstance(J,(int,float)) and isinstance(err,(int,float)) else None
        print('J:',number(J),' | Reward:',number(reward),' | Sai số tham số chuẩn hóa trong train:',number(err))
        print('Actor loss:',number(info.get('actor_loss')),' | Critic loss:',number(info.get('critic_loss')))
        print('Tốc độ:',number(info.get('env_steps_per_second')),'bước/giây | Còn khoảng:',number(info.get('eta_hours')),'giờ')
        print('Đã gặp:',info.get('unique_targets','?'),'mẫu | Thời điểm log:',time.ctime(progress_path.stat().st_mtime) if progress_path.exists() else 'đang khởi tạo')
        print('\n--- Log phiên hiện tại ---')
        log_path=RUN/'training.log'
        if log_path.exists():
            with log_path.open('rb') as stream:
                offset=globals().get('session_log_offset',max(0,log_path.stat().st_size-6000))
                stream.seek(min(offset,log_path.stat().st_size))
                print(stream.read().decode('utf-8',errors='replace')[-6000:])
        status=process.poll() if 'process' in globals() else None
        if 'process' not in globals():
            print('Không có tiến trình thuộc phiên notebook này. Kiểm tra cell Bắt đầu/resume.');break
        print('Trạng thái:', 'Đang chạy' if status is None else f'Đã thoát, mã {status}')
        if status is not None:
            if status==0: print('Train hoàn tất. Chạy cell đánh giá bên dưới.')
            elif status==3: print('Đã dừng an toàn. Có thể resume.')
            else: print('Train báo lỗi. Xem các dòng cuối log trước khi chạy lại.')
            break
        time.sleep(3)
except KeyboardInterrupt:
    print('Đã dừng bảng theo dõi. Train vẫn chạy nền; dùng cell STOP để dừng an toàn.')
