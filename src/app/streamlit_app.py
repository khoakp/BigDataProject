"""Movie recommendation demo backed by the verified REC-01 checkpoint."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import streamlit as st

from src.app.feedback import FeedbackStore
from src.app.rec01_view import (train_catalog, train_history, train_rated_ids,
                                verify_history_source)
from src.models.rec01_inference import load_rec01_inference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
DEFAULT_MOVIES = ROOT.parent / "ml-32m/movies.csv"
DEFAULT_EVENTS = ROOT / "data/processed/canonical_events/canonical_1000000.parquet"
DEFAULT_FEEDBACK = ROOT / "data/processed/s10_feedback/REC01-Kaggle-1M"


@st.cache_resource(show_spinner="Đang tải checkpoint đã xác minh…")
def _model(run_path: str, movies_path: str, report_mtime_ns: int,
           checkpoint_mtime_ns: int, graph_mtime_ns: int, movies_mtime_ns: int):
    # Timestamps invalidate the cache if a local artifact is replaced.
    return load_rec01_inference(Path(run_path), Path(movies_path))


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@st.cache_resource(show_spinner="Đang kiểm tra graph demo…")
def _feedback(run_path: str, events_path: str, state_path: str) -> FeedbackStore:
    return FeedbackStore(Path(run_path), Path(events_path), Path(state_path))


def main() -> None:
    st.set_page_config(page_title="MovieGraph · Gợi ý phim", page_icon="🎬",
                       layout="wide")
    st.title("🎬 MovieGraph")
    st.caption("Top-20 từ LightGCN đã huấn luyện trên canonical prefix 1M của MovieLens 32M")

    run = _path("REC01_RUN_DIR", DEFAULT_RUN)
    movies = _path("MOVIELENS_MOVIES_CSV", DEFAULT_MOVIES)
    events = _path("REC01_EVENTS_PARQUET", DEFAULT_EVENTS)
    feedback_dir = _path("REC01_FEEDBACK_DIR", DEFAULT_FEEDBACK)
    if not run.is_dir() or not movies.is_file() or not events.is_file():
        st.error("Thiếu REC-01 artifact, movies.csv hoặc canonical events. Xem README để chuẩn bị dữ liệu.")
        st.stop()

    try:
        report = run / "report.json"
        selected = json.loads(report.read_text(encoding="utf-8"))
        checkpoint = run / selected["selected_checkpoint_file"]
        model = _model(str(run), str(movies), report.stat().st_mtime_ns,
                       checkpoint.stat().st_mtime_ns,
                       (run / "model_graph/manifest.json").stat().st_mtime_ns,
                       movies.stat().st_mtime_ns)
        verify_history_source(events, model.report["source"]["sha256"])
        feedback = _feedback(str(run), str(events), str(feedback_dir))
        state = feedback.status()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Không thể tải artifact đã xác minh: {exc}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Graph demo version", state["canonical_graph_version"])
    col2.metric("Model graph version", state["model_graph_version"])
    col3.metric("Checkpoint step", state["checkpoint_step"])
    col4.metric("Batch kể từ checkpoint", state["staleness_batches"])
    st.info("Graph demo nhận rating mới theo batch. Checkpoint LightGCN vẫn ở graph version "
            f"{model.graph_version}; gợi ý chỉ đổi khi model refresh được thực hiện thực sự.")

    with st.sidebar:
        st.header("Chọn người dùng")
        mode = st.radio("Luồng demo", ("Người dùng đã có", "Người dùng mới"))
        if mode == "Người dùng đã có":
            users = sorted(model.user_nodes)
            preferred = 173074 if 173074 in model.user_nodes else users[0]
            user_id = st.selectbox("User ID trong train graph", users,
                                   index=users.index(preferred))
        else:
            user_id = int(st.number_input("User ID mới",
                                          min_value=max(model.user_nodes) + 1,
                                          value=max(model.user_nodes) + 1, step=1))
            st.caption("User mới không có embedding; hệ thống dùng Popularity từ train graph.")
        st.caption(f"Run: {model.report['run_id']}")
        st.caption(f"Train cutoff: {model.report['protocol']['train_end_ts']}")

    result = model.recommend(user_id)
    if result["method"] == "lightgcn":
        st.subheader(f"Top-20 cho user {user_id} · LightGCN")
        st.caption(f"Đã ẩn {result['train_seen_movies']} phim user từng đánh giá positive trong train graph.")
    else:
        st.subheader("Top-20 phổ biến · dành cho người dùng mới")
        st.caption("Điểm là số người dùng positive trong train; không phải điểm LightGCN.")

    ranked = [{"Hạng": item["rank"], "Phim": item["title"] or "Chưa có tên phim",
               "Movie ID": item["movie_raw_id"], "Thể loại": item["genres"] or "Chưa có thể loại",
               "Điểm": round(item["score"], 5)} for item in result["items"]]
    if ranked:
        st.dataframe(ranked, hide_index=True, width="stretch")
    else:
        st.warning("Không còn phim chưa xem trong train catalog cho user này.")
    coverage = result["metadata_coverage"]
    st.caption(f"Metadata: {coverage['metadata_found']}/{coverage['rows']} phim có thông tin catalog.")

    history_tab, catalog_tab = st.tabs(("Lịch sử train", "Danh mục phim ứng viên"))
    with history_tab:
        history = train_history(events, user_id, model.report["protocol"]["train_end_ts"],
                                model.movies, limit=100)
        st.write(f"{history['total_train_ratings']} ratings trước cutoff; "
                 f"{history['positive_train_ratings']} ratings đạt ngưỡng 4,0. "
                 f"Hiển thị tối đa {history['limit']} dòng mới nhất.")
        if history["rows"]:
            st.dataframe(history["rows"], hide_index=True, width="stretch")
        else:
            st.caption("Chưa có lịch sử trong train snapshot.")
    with catalog_tab:
        st.caption("Chỉ gồm phim có positive train edge; đây là candidate universe của REC-01.")
        st.dataframe(train_catalog(model.movie_nodes, model.popularity, model.movies),
                     hide_index=True, width="stretch")

    st.subheader("Đánh giá phim và cập nhật graph demo")
    st.caption("Thêm tối đa 20 rating vào hàng chờ, rồi áp dụng thành một batch. "
               "Rating từ 4,0 tạo cạnh positive; rating thấp hơn vẫn được lưu nhưng không tạo cạnh.")
    queue = st.session_state.setdefault("feedback_queue", [])
    request_id = st.session_state.setdefault("feedback_request_id", str(uuid.uuid4()))
    already_rated = train_rated_ids(events, user_id,
                                    model.report["protocol"]["train_end_ts"])
    already_rated.update(row["movie_raw_id"] for row in feedback.applied_ratings()
                         if row["user_raw_id"] == user_id)
    already_rated.update(row["movie_raw_id"] for row in queue
                         if row["user_raw_id"] == user_id)
    eligible = [movie_id for movie_id in sorted(model.movie_nodes)
                if movie_id not in already_rated]
    if eligible and len(queue) < 20:
        with st.form("add_rating", clear_on_submit=False):
            movie_id = st.selectbox("Phim chưa đánh giá (Movie ID)", eligible)
            st.caption(model.movies.get(movie_id, (None, None))[0] or "Chưa có tên phim")
            rating = st.slider("Rating", min_value=0.5, max_value=5.0,
                               value=4.0, step=0.5)
            add = st.form_submit_button("Thêm vào batch")
        if add:
            queue.append({"user_raw_id": user_id, "movie_raw_id": movie_id,
                          "rating": rating})
            st.rerun()
    elif not eligible:
        st.caption("User này đã đánh giá toàn bộ phim ứng viên trong train catalog.")

    if queue:
        st.write(f"Đang chờ áp dụng: {len(queue)} rating")
        st.dataframe([{"User ID": row["user_raw_id"], "Movie ID": row["movie_raw_id"],
                       "Phim": model.movies.get(row["movie_raw_id"], (None, None))[0],
                       "Rating": row["rating"]} for row in queue],
                     hide_index=True, width="stretch")
        if st.button("Áp dụng batch vào graph", type="primary"):
            try:
                result = feedback.apply_ratings(queue, request_id)
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Không áp dụng được batch: {exc}")
            else:
                st.session_state["feedback_queue"] = []
                st.session_state["feedback_request_id"] = str(uuid.uuid4())
                st.session_state["feedback_last_result"] = result
                st.rerun()
        if st.button("Xóa hàng chờ"):
            st.session_state["feedback_queue"] = []
            st.session_state["feedback_request_id"] = str(uuid.uuid4())
            st.rerun()
    if "feedback_last_result" in st.session_state:
        latest = st.session_state["feedback_last_result"]
        st.success(f"Batch {latest['batch_id']}: {latest['status']}; graph version "
                   f"{latest['graph_version']}; thêm {latest['inserted_edges']} cạnh positive.")
    recent = [row for row in feedback.applied_ratings()
              if row["user_raw_id"] == user_id]
    if recent:
        st.write(f"Rating mới của user {user_id} trong graph demo")
        st.dataframe(recent[-20:][::-1], hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
