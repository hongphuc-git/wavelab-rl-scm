# Bảng cố định: cập nhật tại chỗ, không clear_output toàn cell.
import time,json,html
from IPython.display import display,HTML
panel=display(HTML('<p>Đang đọc tiến độ...</p>'),display_id=True)
last_view=None
monitor_log=RUN/'training.log'
# Nếu cell launch cũ không ghi offset, chỉ đọc phiên OUTPUT gần nhất.
if 'session_log_offset' not in globals():
    tail=monitor_log.read_bytes() if monitor_log.exists() else b''
    marker=tail.rfind(b'OUTPUT:')
    session_log_offset=marker if marker>=0 else max(0,len(tail)-4000)
try:
    while True:
        progress_path=RUN/'progress.json'
        try: info=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        except (OSError,ValueError): info={}
        status=process.poll() if 'process' in globals() else 'unknown'
        text=''
        if monitor_log.exists():
            with monitor_log.open('rb') as stream:
                stream.seek(min(session_log_offset,monitor_log.stat().st_size))
                text=stream.read().decode('utf-8',errors='replace')[-5000:]
        step=int(info.get('step',0));pct=min(100,100*step/TOTAL_STEPS)
        def fmt(value):return f'{value:.4f}' if isinstance(value,(int,float)) else '—'
        state='Đang chạy' if status is None else 'Không có tiến trình trong phiên notebook này' if status=='unknown' else 'Hoàn tất' if status==0 else 'Đã dừng an toàn' if status==3 else f'Đã dừng do lỗi · mã {status}'
        J=info.get('mean_J');error=info.get('mean_param_error')
        reward=-J-.01-2*error if isinstance(J,(int,float)) and isinstance(error,(int,float)) else None
        view=(step,info.get('critic_loss'),status)
        if view!=last_view:
            panel.update(HTML(f"""<div style="border:1px solid #438b91;border-radius:12px;padding:16px">
<h3>EfficientNet V2-L epoch {EPOCH} + SAC</h3>
<progress value="{step}" max="{TOTAL_STEPS}" style="width:100%;height:24px"></progress>
<p><b>{step:,} / {TOTAL_STEPS:,} bước · {pct:.1f}% · {html.escape(state)}</b></p>
<p>J: {fmt(J)} · Reward: {fmt(reward)} · Sai số chuẩn hóa trong train: {fmt(error)}</p>
<p>Actor loss: {fmt(info.get('actor_loss'))} · Critic loss: {fmt(info.get('critic_loss'))}</p>
<p>Tốc độ: {fmt(info.get('env_steps_per_second'))} bước/giây · Còn khoảng {fmt(info.get('eta_hours'))} giờ</p>
<p>Log kỹ thuật được lưu trong training.log; không hiển thị trên bảng.</p>
</div>"""))
            last_view=view
        if status is not None:break
        time.sleep(3)
except KeyboardInterrupt:
    print('Dừng theo dõi; train vẫn chạy nền. Dùng STOP để dừng train an toàn.')
