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

SUN_COLOR = (1.0, 0.95, 0.8)  # Màu vàng nhạt
SUN_AZIMUTH = 160.0     # Mặt trời bên trái, hơi lệch về phía tòa nhà
SUN_ELEVATION = 45.0    # Thấp hơn -> bóng dài, dễ nhìn thấy hơn
SUN_INTENSITY = 1.0
AMBIENT_COLOR = (0.22, 0.26, 0.40)

# --- Shadow mapping ---
SHADOW_MAP_SIZE = 2048
SHADOW_RADIUS = 80.0        # Shadow map phủ hình vuông cạnh 2R mét, tâm là người chơi
SHADOW_DISTANCE = 150.0     # Khoảng cách từ "mắt" mặt trời tới tâm (phải lớn hơn vật cao nhất)
SHADOW_BIAS = 0.0002        # Bias độ sâu (đơn vị NDC 0..1)
SHADOW_NORMAL_OFFSET = 1.0  # Dịch điểm theo pháp tuyến (đơn vị: số texel)

# --- Người chơi & va chạm ---
PLAYER_RADIUS = 0.35     # nửa bề rộng hộp va chạm (m)
PLAYER_HEAD_GAP = 0.2    # hộp va chạm cao hơn mắt bao nhiêu (m)

# --- Guided Tour ---
TOUR_SPEED = 3.5         # m/s dọc theo đường cong
TOUR_LOOK_AHEAD = 6.0    # nhìn về điểm cách phía trước bao nhiêu mét
TOUR_LOOK_SMOOTH = 3.0   # hệ số làm mượt hướng nhìn (lớn = bám nhanh)
# (x, y, z). Các điểm này tránh 4 tòa nhà hiện tại; hãy chỉnh theo bản đồ của bạn.
TOUR_WAYPOINTS = [
    (0, 1.7, 15), (-15, 1.7, 0), (-15, 1.7, -22), (0, 4.0, -26),
    (15, 1.7, -22), (15, 1.7, 0), (10, 1.7, 20),
]