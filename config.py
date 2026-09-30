"""Cấu hình toàn cục cho dự án IU Campus Virtual Tour.

Gom mọi hằng số vào một nơi để dễ chỉnh và dễ trích dẫn vào báo cáo.
"""

# --- Cửa sổ ---
WINDOW_TITLE = "IU Campus Virtual Tour"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VSYNC = True            # Đồng bộ với tần số quét màn hình -> tránh xé hình
MSAA_SAMPLES = 4        # Khử răng cưa 4x (đặt 0 để tắt nếu GPU yếu)

# --- OpenGL ---
GL_MAJOR = 3
GL_MINOR = 3            # Core profile 3.3 -> chạy tốt trên Windows/Linux/macOS

# PyOpenGL mặc định gọi glGetError sau MỌI lệnh GL -> rất chậm.
# Bật True khi debug, để False khi đo FPS / nộp bản cuối.
GL_ERROR_CHECKING = True

# --- Render ---
CLEAR_COLOR = (0.53, 0.81, 0.92, 1.0)   # Xanh da trời nhạt (tạm, sẽ thay bằng Skybox ở Phase 3)
FPS_CAP = 0             # 0 = không giới hạn (khi VSYNC bật thì màn hình tự giới hạn)
FPS_TITLE_INTERVAL = 0.5  # Giây giữa hai lần cập nhật FPS lên tiêu đề cửa sổ
