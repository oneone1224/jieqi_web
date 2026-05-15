# -*- coding: utf-8 -*-
"""
24节气导引动作评估Web应用 - 整合版
改动：MMPose → MediaPipe，新增摄像头录制，路径改为相对路径
"""

import streamlit as st
import os
import json
import tempfile
import threading
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# ============ 路径配置（相对路径，适配 Streamlit Cloud）============
FEATURE_ROOT  = r'F:\paper_material\feature_data_wjz'
SKELETON_ROOT = r'F:\paper_material\skeleton_data_wjz'
VIDEO_ROOT    = r'F:\paper_material\vedios_data_wjz'   # 注意：视频在这个根目录下，每个mp4直接在里面
TEMP_ROOT     = r'F:\paper_material\temp_uploads_wjz'
os.makedirs(TEMP_ROOT, exist_ok=True)

# ============ 节气映射 ============
JIEQI_LIST = [
    '立春', '雨水', '惊蛰', '春分', '清明', '谷雨',
    '立夏', '小满', '芒种', '夏至', '小暑', '大暑',
    '立秋', '处暑', '白露', '秋分', '寒露', '霜降',
    '立冬', '小雪', '大雪', '冬至', '小寒', '大寒'
]

JIEQI_EN = {
    '立春': 'lichun',   '雨水': 'yushui',    '惊蛰': 'jingzhe',
    '春分': 'chunfen',  '清明': 'qingming',  '谷雨': 'guyu',
    '立夏': 'lixia',    '小满': 'xiaoman',   '芒种': 'mangzhong',
    '夏至': 'xiazhi',   '小暑': 'xiaoshu',   '大暑': 'dashu',
    '立秋': 'liqiu',    '处暑': 'chushu',    '白露': 'bailu',
    '秋分': 'qiufen',   '寒露': 'hanlu',     '霜降': 'shuangjiang',
    '立冬': 'lidong',   '小雪': 'xiaoxue',   '大雪': 'daxue',
    '冬至': 'dongzhi',  '小寒': 'xiaohan',   '大寒': 'dahan'
}

DIMENSION_NAMES = {
    'dtw':      {'cn': 'DTW时序相似度',  'en': 'DTW Temporal Similarity', 'icon': '⏱️', 'desc': '动作节奏和时序匹配度'},
    'angle':    {'cn': '关节角度相似度', 'en': 'Joint Angle Similarity',  'icon': '📐', 'desc': '肘、膝、肩等关节角度'},
    'range':    {'cn': '运动幅度相似度', 'en': 'Motion Range Similarity', 'icon': '📏', 'desc': '手臂、腿部的运动范围'},
    'velocity': {'cn': '速度匹配度',    'en': 'Velocity Matching',       'icon': '⚡', 'desc': '动作速度一致性'}
}

GRADE_NAMES = {'优秀': 'Excellent', '良好': 'Good', '中等': 'Fair', '及格': 'Pass', '需要改进': 'Need Improvement'}

# ============ MediaPipe → COCO 17点 映射 ============
# MediaPipe Pose 有33个点，取其中17个对应 COCO 格式
MP_TO_COCO = {
    0: 0,   # nose
    2: 1,   # left_eye
    5: 2,   # right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
}

# ============ MediaPipe 骨架提取 ============

def extract_skeleton_mediapipe(frames):
    """
    输入：帧列表（BGR numpy数组）
    输出：skeleton_data 字典，格式与 MMPose 版本完全一致
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    skeleton_sequence = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks is None:
                continue

            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark

            # 转为 COCO 17点格式
            keypoints = np.zeros((17, 2))
            scores    = np.zeros(17)

            for mp_idx, coco_idx in MP_TO_COCO.items():
                lm = landmarks[mp_idx]
                keypoints[coco_idx] = [lm.x * w, lm.y * h]
                scores[coco_idx]    = lm.visibility

            avg_score = float(scores.mean())
            if avg_score < 0.3:
                continue

            skeleton_sequence.append({
                'frame_id':     f'frame_{i:04d}.jpg',
                'frame_number': i,
                'keypoints':    keypoints.tolist(),
                'scores':       scores.tolist(),
                'avg_score':    avg_score
            })

    return {
        'jieqi_name':      'user_input',
        'total_frames':    len(skeleton_sequence),
        'skeleton_sequence': skeleton_sequence
    }


# ============ 特征提取（与原版一致）============

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def extract_features_from_skeleton(skeleton_data):
    """从 skeleton_data 字典直接提取特征，不依赖文件路径"""
    seq = skeleton_data['skeleton_sequence']

    frame_angles = []
    for fd in seq:
        kpts = np.array(fd['keypoints'])
        angles = {}
        if len(kpts) > 9:
            angles['left_elbow']  = calculate_angle(kpts[5], kpts[7],  kpts[9])
            angles['right_elbow'] = calculate_angle(kpts[6], kpts[8],  kpts[10])
        if len(kpts) > 15:
            angles['left_knee']   = calculate_angle(kpts[11], kpts[13], kpts[15])
            angles['right_knee']  = calculate_angle(kpts[12], kpts[14], kpts[16])
        frame_angles.append({
            'frame_id':     fd['frame_id'],
            'frame_number': fd['frame_number'],
            'angles':       angles
        })

    motion_ranges = {}
    for i in range(17):
        positions = [np.array(fd['keypoints'])[i] for fd in seq if i < len(fd['keypoints'])]
        if positions:
            pos = np.array(positions)
            xr = pos[:, 0].max() - pos[:, 0].min()
            yr = pos[:, 1].max() - pos[:, 1].min()
            motion_ranges[f'keypoint_{i}'] = {
                'x_range': float(xr),
                'y_range': float(yr),
                'total_range': float(np.sqrt(xr**2 + yr**2))
            }

    velocities = {}
    for i in range(17):
        traj = [np.array(fd['keypoints'])[i] for fd in seq if i < len(fd['keypoints'])]
        if len(traj) > 1:
            traj = np.array(traj)
            vel = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            velocities[f'keypoint_{i}'] = vel.tolist()

    return {
        'jieqi_name':   'user_input',
        'total_frames': len(seq),
        'frame_angles': frame_angles,
        'motion_ranges': motion_ranges,
        'velocities':   velocities,
        'key_frames':   []
    }


# ============ 视频 → 帧列表（10fps）============

def video_to_frames(video_path, fps=10):
    """读取视频，返回 BGR 帧列表"""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval  = max(1, int(video_fps / fps))
    frames = []
    count  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


# ============ 评估器（与原版完全一致）============

class ActionEvaluator:

    def __init__(self, template_features, template_skeleton=None):
        self.template_features = template_features
        self.template_skeleton = template_skeleton
        self.weights = {
            'dtw':      0.30,
            'angle':    0.50,
            'range':    0.15,
            'velocity': 0.05
        }

    def calculate_dtw_similarity(self, user_traj, temp_traj):
        u = np.array(user_traj)
        t = np.array(temp_traj)
        if len(u) == 0 or len(t) == 0:
            return 0
        dist, _ = fastdtw(u, t, dist=euclidean)
        norm = dist / max(len(u), len(t))
        return max(0, min(100, 100 * np.exp(-norm / 300)))

    def calculate_angle_similarity(self, user_angles, template_angles):
        names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
        sims  = []
        for name in names:
            us = [f['angles'].get(name, np.nan) for f in user_angles]
            ts = [f['angles'].get(name, np.nan) for f in template_angles]
            us = [a for a in us if not np.isnan(a)]
            ts = [a for a in ts if not np.isnan(a)]
            if not us or not ts:
                continue
            dist, _ = fastdtw(np.array(us).reshape(-1,1), np.array(ts).reshape(-1,1), dist=euclidean)
            norm = dist / max(len(us), len(ts))
            sims.append(100 * np.exp(-norm / 40))
        return float(np.mean(sims)) if sims else 0

    def calculate_range_similarity(self, user_ranges, template_ranges):
        kpts = ['keypoint_9', 'keypoint_10', 'keypoint_5', 'keypoint_6', 'keypoint_11', 'keypoint_12']
        sims = []
        for kpt in kpts:
            if kpt not in user_ranges or kpt not in template_ranges:
                continue
            ur = user_ranges[kpt]['total_range']
            tr = template_ranges[kpt]['total_range']
            if tr > 0:
                ratio = ur / tr
                sim = 100 * (1 - abs(ratio-1)/0.8) if 0.3 <= ratio <= 2.0 else 100 * np.exp(-abs(ratio-1)/2.0)
                sims.append(max(0, sim))
        return float(np.mean(sims)) if sims else 0

    def calculate_velocity_similarity(self, user_vel, temp_vel):
        u = np.array(user_vel.get('keypoint_9', []))
        t = np.array(temp_vel.get('keypoint_9', []))
        if len(u) == 0 or len(t) == 0:
            return 70
        ratio = u.mean() / (t.mean() + 1e-6)
        return max(0, min(100, 100 * np.exp(-abs(np.log(ratio + 1e-6)) / 3.0)))

    def evaluate(self, user_features, user_skeleton=None):
        scores = {}

        # DTW（使用真实轨迹，与原版一致）
        if user_skeleton and self.template_skeleton:
            ut = self._extract_traj(user_skeleton, 9)
            tt = self._extract_traj(self.template_skeleton, 9)
            scores['dtw'] = self.calculate_dtw_similarity(ut, tt)
        else:
            scores['dtw'] = 70

        scores['angle']    = self.calculate_angle_similarity(user_features['frame_angles'], self.template_features['frame_angles'])
        scores['range']    = self.calculate_range_similarity(user_features['motion_ranges'], self.template_features['motion_ranges'])
        scores['velocity'] = self.calculate_velocity_similarity(user_features['velocities'], self.template_features['velocities'])

        total = sum(scores[d] * self.weights[d] for d in scores)
        return {
            'total_score':       total,
            'dimension_scores':  scores,
            'weights':           self.weights,
            'suggestions':       self._suggestions(scores),
            'grade':             self._grade(total)
        }

    def _extract_traj(self, skeleton_data, idx):
        return np.array([
            np.array(f['keypoints'])[idx]
            for f in skeleton_data['skeleton_sequence']
            if idx < len(f['keypoints'])
        ])

    def _suggestions(self, scores):
        items = [
            ('dtw',      80, "动作节奏控制良好 Good rhythm control",          "注意动作的节奏和速度 Pay attention to rhythm"),
            ('angle',    80, "关节角度标准 Standard joint angles",             "调整关节角度 Adjust joint angles"),
            ('range',    75, "动作幅度良好 Good motion range",                 "增加动作幅度 Increase motion range"),
            ('velocity', 70, "速度控制良好 Good velocity control",             "保持匀速进行 Maintain constant speed"),
        ]
        result = []
        for dim, threshold, good, bad in items:
            if scores[dim] >= threshold:
                result.append(("✅", "优点 Strength", good))
            else:
                result.append(("⚠️", "建议 Suggestion", bad))
        return result

    def _grade(self, score):
        if score >= 90: return "优秀"
        if score >= 80: return "良好"
        if score >= 70: return "中等"
        if score >= 60: return "及格"
        return "需要改进"


# ============ 核心评估流程 ============

def run_evaluation(frames, jieqi_en, status_placeholder):
    """给定帧列表，跑完整评估流程，返回 result 字典"""

    status_placeholder.info("🦴 正在提取骨架关键点（MediaPipe）... Extracting skeleton...")
    skeleton_data = extract_skeleton_mediapipe(frames)
    n = skeleton_data['total_frames']
    if n == 0:
        raise ValueError("未能检测到人体骨架，请确保画面中人体清晰完整。")
    status_placeholder.success(f"✅ 已提取 {n} 帧骨架")

    status_placeholder.info("📊 正在提取动作特征... Extracting features...")
    user_features = extract_features_from_skeleton(skeleton_data)
    status_placeholder.success("✅ 特征提取完成")

    status_placeholder.info("🎯 正在评估动作... Evaluating...")
    feat_path = os.path.join(FEATURE_ROOT, f'{jieqi_en}_features.json')
    skel_path = os.path.join(SKELETON_ROOT, f'{jieqi_en}_skeleton.json')

    with open(feat_path, 'r', encoding='utf-8') as f:
        temp_feat = json.load(f)
    with open(skel_path, 'r', encoding='utf-8') as f:
        temp_skel = json.load(f)

    evaluator = ActionEvaluator(temp_feat, temp_skel)
    result = evaluator.evaluate(user_features, skeleton_data)
    status_placeholder.success("✅ 评估完成！Assessment completed!")
    return result


# ============ 结果展示 ============

def show_results(result):
    st.markdown("---")
    st.success("✅ 评估完成！Assessment Completed!")
    st.markdown("### 📊 评估结果 Assessment Results")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("总分 Total Score", f"{result['total_score']:.1f}")
    with c2:
        grade_cn = result['grade']
        st.metric("等级 Grade", f"{grade_cn} {GRADE_NAMES.get(grade_cn, '')}")
    with c3:
        st.metric("进步空间 Room for Improvement", f"{100 - result['total_score']:.1f}分")

    st.markdown("#### 📈 各维度得分 Dimension Scores")
    for dim, score in result['dimension_scores'].items():
        weight = result['weights'][dim]
        info   = DIMENSION_NAMES[dim]
        col1, col2 = st.columns([3, 2])
        with col1:
            st.progress(score / 100,
                text=f"{info['icon']} **{info['cn']}** ({info['en']}): **{score:.1f}分** (权重 {weight*100:.0f}%)")
        with col2:
            st.caption(f"💡 {info['desc']}")

    st.markdown("#### 💡 改进建议 Suggestions")
    for icon, label, text in result['suggestions']:
        if "优点" in label:
            st.success(f"{icon} **{label}:** {text}")
        else:
            st.warning(f"{icon} **{label}:** {text}")


# ============ Streamlit 页面 ============

st.set_page_config(page_title="24节气导引动作评估系统", page_icon="🧘", layout="wide")
st.title("🧘 二十四节气中医导引动作智能评估系统")
st.markdown("**24 Solar Terms TCM Guiding Exercise Intelligent Assessment System**")
st.markdown("---")

# 侧边栏
st.sidebar.header("选择节气 Select Solar Term")
selected_jieqi = st.sidebar.selectbox("选择要练习的节气", JIEQI_LIST)
jieqi_en = JIEQI_EN[selected_jieqi]
st.sidebar.markdown("---")
st.sidebar.info(f"**当前节气 Current:** {selected_jieqi}")

# 主界面：左列标准视频 / 右列输入
col1, col2 = st.columns([1, 1])

with col1:
    st.header(f"📹 {selected_jieqi} 标准动作")
    video_path = os.path.join(VIDEO_ROOT, f'{jieqi_en}.mp4')
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info("标准视频文件未找到（可将 mp4 放入 data/videos/）")

with col2:
    st.header("🎬 上传 / 录制您的练习视频")
    st.warning(f"⚠️ 请确保上传或录制的是 **{selected_jieqi}** 的动作视频！")

    tab_upload, tab_camera = st.tabs(["📤 上传视频", "📷 摄像头录制"])

    # ── Tab 1：上传视频 ──
    with tab_upload:
        uploaded = st.file_uploader(
            "选择视频文件（MP4 / AVI / MOV）",
            type=['mp4', 'avi', 'mov']
        )
        if uploaded:
            st.video(uploaded)
            if st.button("🎯 开始评估（上传视频）", type="primary", key="btn_upload"):
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                with st.spinner("处理中，请稍候..."):
                    status = st.empty()
                    try:
                        status.info("📹 正在提取视频帧... Extracting frames...")
                        frames = video_to_frames(tmp_path, fps=10)
                        status.success(f"✅ 已提取 {len(frames)} 帧")
                        result = run_evaluation(frames, jieqi_en, status)
                        st.session_state.result    = result
                        st.session_state.evaluated = True
                    except Exception as e:
                        st.error(f"处理出错：{e}")
                        import traceback; st.code(traceback.format_exc())
                os.unlink(tmp_path)

    # ── Tab 2：摄像头录制 ──
    with tab_camera:
        st.info("💡 点击「开始录制」后，做完整套动作，再点「停止并评估」")

        # 用 session_state 管理录制状态
        if 'recording'       not in st.session_state: st.session_state.recording       = False
        if 'recorded_frames' not in st.session_state: st.session_state.recorded_frames = []

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🔴 开始录制", disabled=st.session_state.recording, key="btn_start"):
                st.session_state.recording       = True
                st.session_state.recorded_frames = []
                st.rerun()

        with col_btn2:
            if st.button("⏹ 停止并评估", disabled=not st.session_state.recording, key="btn_stop"):
                st.session_state.recording = False
                st.rerun()

        # 摄像头画面 + 录制帧采集
        if st.session_state.recording:
            st.warning("🔴 录制中... 做完动作后点「停止并评估」")
            camera_image = st.camera_input("摄像头画面（每次快照会被采集）", key="cam")

            if camera_image is not None:
                # 将快照解码为 BGR 帧
                file_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    st.session_state.recorded_frames.append(frame)
                    st.caption(f"已采集 {len(st.session_state.recorded_frames)} 帧，继续做动作...")

        elif len(st.session_state.recorded_frames) > 0 and not st.session_state.get('evaluated', False):
            frames = st.session_state.recorded_frames
            st.success(f"✅ 录制完成，共 {len(frames)} 帧，正在评估...")

            with st.spinner("处理中..."):
                status = st.empty()
                try:
                    result = run_evaluation(frames, jieqi_en, status)
                    st.session_state.result    = result
                    st.session_state.evaluated = True
                    st.session_state.recorded_frames = []  # 清空
                except Exception as e:
                    st.error(f"处理出错：{e}")
                    import traceback; st.code(traceback.format_exc())

        elif not st.session_state.recording and not st.session_state.recorded_frames:
            st.info("👆 点击「开始录制」开始采集动作帧")

# ============ 显示结果 ============
if st.session_state.get('evaluated', False):
    show_results(st.session_state.result)

    if st.button("🔄 重新评估 Reset"):
        st.session_state.evaluated = False
        st.session_state.result    = None
        st.rerun()

st.markdown("---")
st.caption("24节气中医导引动作智能评估系统 v2.0 | TCM 24 Solar Terms Exercise Assessment System")
