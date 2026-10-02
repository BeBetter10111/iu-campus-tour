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


# --- Camera ---
FOV = 60.0            # Góc nhìn dọc (độ)
NEAR_PLANE = 0.1
FAR_PLANE = 500.0
EYE_HEIGHT = 1.7      # Chiều cao mắt người (m)
WALK_SPEED = 4.0      # m/s
RUN_SPEED = 8.0       # m/s (giữ Shift)
MOUSE_SENSITIVITY = 0.1  # độ / pixel

# --- Đường dẫn assets ---
from pathlib import Path
ASSET_DIR = Path(__file__).resolve().parent / "assets"
TEXTURE_DIR = ASSET_DIR / "textures"
MODEL_DIR = ASSET_DIR / "models"
SKYBOX_DIR = ASSET_DIR / "skybox"


SUN_AZIMUTH = 160.0     # Mặt trời bên trái, hơi lệch về phía tòa nhà
SUN_ELEVATION = 35.0    # Thấp hơn -> bóng dài, dễ nhìn thấy hơn
SUN_INTENSITY = 1.0
AMBIENT_COLOR = (0.25, 0.28, 0.35)

# --- Shadow mapping ---
SHADOW_MAP_SIZE = 2048
SHADOW_RADIUS = 80.0        # Shadow map phủ hình vuông cạnh 2R mét, tâm là người chơi
SHADOW_DISTANCE = 150.0     # Khoảng cách từ "mắt" mặt trời tới tâm (phải lớn hơn vật cao nhất)
SHADOW_BIAS = 0.0002        # Bias độ sâu (đơn vị NDC 0..1)
SHADOW_NORMAL_OFFSET = 1.0  # Dịch điểm theo pháp tuyến (đơn vị: số texel)