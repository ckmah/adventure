import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import spatialdata as sd
    import scanpy as sc
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import altair as alt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    import wigglystuff
    from wigglystuff import LandmarksWidget, ParallelCoordinates

    return (
        LandmarksWidget,
        ParallelCoordinates,
        StandardScaler,
        alt,
        gpd,
        matplotlib,
        mo,
        np,
        pd,
        plt,
        sc,
        sd,
        sns,
    )


@app.cell
def _(matplotlib, mo):
    theme = mo.app_meta().theme
    matplotlib.style.use("dark_background" if theme == "dark" else "default")
    return (theme,)


@app.cell
def _(gpd, np, pd):
    from shapely.geometry import LineString, Point, Polygon

    def cardinal_sample(vertices, tension=0.0, n_per_seg=20, closed=False):
        pts = [(float(x), float(y)) for x, y in vertices]
        if closed:
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                return pts
            n = len(pts)

            def at(i):
                return pts[i % n]

            n_seg = n
        else:
            if len(pts) < 2:
                return pts
            if len(pts) == 2:
                return pts
            n = len(pts)
            ext = [
                (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1]),
                *pts,
                (2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1]),
            ]

            def at(i):
                return ext[i + 1]

            n_seg = n - 1

        s = (1.0 - max(0.0, min(1.0, float(tension)))) / 2.0
        out = []
        for i in range(n_seg):
            p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
            m1x = s * (p2[0] - p0[0])
            m1y = s * (p2[1] - p0[1])
            m2x = s * (p3[0] - p1[0])
            m2y = s * (p3[1] - p1[1])
            for j in range(n_per_seg):
                u = j / n_per_seg
                u2 = u * u
                u3 = u2 * u
                h00 = 2 * u3 - 3 * u2 + 1
                h10 = u3 - 2 * u2 + u
                h01 = -2 * u3 + 3 * u2
                h11 = u3 - u2
                out.append(
                    (
                        h00 * p1[0] + h10 * m1x + h01 * p2[0] + h11 * m2x,
                        h00 * p1[1] + h10 * m1y + h01 * p2[1] + h11 * m2y,
                    )
                )
        out.append(at(n_seg if closed else n - 1))
        return out

    def landmark_geoms(landmarks):
        geoms = {}
        for lm in landmarks:
            if lm.get("hidden"):
                continue
            kind = lm.get("type", "point")
            if kind == "gradient":
                kind = "spline"
            verts = lm.get("vertices") or []
            data_pts = [(float(v[0]), float(v[1])) for v in verts]
            lid = str(lm.get("id"))
            if kind == "point" and data_pts:
                geoms[lid] = ("point", Point(data_pts[0]))
            elif kind == "line" and len(data_pts) >= 2:
                geoms[lid] = ("line", LineString(data_pts))
            elif kind == "spline":
                sampled = cardinal_sample(data_pts, float(lm.get("tension") or 0.0))
                if len(sampled) >= 2:
                    geoms[lid] = ("spline", LineString(sampled))
            elif kind == "shape":
                sampled = cardinal_sample(
                    data_pts, float(lm.get("tension") or 0.0), closed=True
                )
                if len(sampled) >= 3:
                    ring = list(sampled)
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])
                    geoms[lid] = ("shape", Polygon(ring))
        return geoms

    def distances(x, y, groups, landmarks, indices):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        indices = np.asarray(indices, dtype=int)
        points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        rows = []
        for lid, (ltype, geom) in landmark_geoms(landmarks).items():
            dist = points.distance(geom).to_numpy()
            for i in indices:
                rows.append(
                    {
                        "point_index": int(i),
                        "landmark_id": lid,
                        "landmark_type": ltype,
                        "group": groups[i],
                        "distance": float(dist[i]),
                    }
                )
        return pd.DataFrame(rows)

    def composition(x, y, groups, landmarks, indices):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        indices = np.asarray(indices, dtype=int)
        points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        cand = np.zeros(len(x), dtype=bool)
        cand[indices] = True
        rows = []
        for lid, (ltype, geom) in landmark_geoms(landmarks).items():
            if ltype != "shape":
                continue
            mask = cand & points.intersects(geom).to_numpy()
            subset = groups[mask]
            n = int(mask.sum())
            if n == 0:
                continue
            values, counts = np.unique(subset, return_counts=True)
            for value, count in zip(values, counts, strict=True):
                rows.append(
                    {
                        "landmark_id": lid,
                        "group": value,
                        "count": int(count),
                        "proportion": float(count) / n,
                        "n_total": n,
                    }
                )
        return pd.DataFrame(rows)

    def profile(x, y, groups, landmarks, indices, n=40, radius=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        indices = np.asarray(indices, dtype=int)
        points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        cand = np.zeros(len(x), dtype=bool)
        cand[indices] = True
        if radius is None:
            radius = 0.05 * max(float(np.ptp(x) or 1.0), float(np.ptp(y) or 1.0))
        all_groups = [g for g in pd.unique(groups) if g is not None]
        rows = []
        for lid, (ltype, geom) in landmark_geoms(landmarks).items():
            if ltype not in ("line", "spline") or geom.geom_type != "LineString":
                continue
            for i in range(n):
                s = i / max(n - 1, 1)
                pt = geom.interpolate(s, normalized=True)
                neighborhood = (points.distance(pt) <= radius).to_numpy() & cand
                local_groups = groups[neighborhood]
                local_dist = points.distance(pt).to_numpy()[neighborhood]
                n_pts = int(neighborhood.sum())
                mean_dist = float(local_dist.mean()) if n_pts else float("nan")
                if n_pts == 0:
                    for g in all_groups:
                        rows.append(
                            {
                                "landmark_id": lid,
                                "landmark_type": ltype,
                                "s": s,
                                "x": float(pt.x),
                                "y": float(pt.y),
                                "n_points": 0,
                                "mean_dist": mean_dist,
                                "group": g,
                                "proportion": float("nan"),
                            }
                        )
                    continue
                values, counts = np.unique(local_groups, return_counts=True)
                freq = {v: c / n_pts for v, c in zip(values, counts, strict=True)}
                for g in all_groups:
                    rows.append(
                        {
                            "landmark_id": lid,
                            "landmark_type": ltype,
                            "s": s,
                            "x": float(pt.x),
                            "y": float(pt.y),
                            "n_points": n_pts,
                            "mean_dist": mean_dist,
                            "group": g,
                            "proportion": float(freq.get(g, 0.0)),
                        }
                    )
        return pd.DataFrame(rows)

    return composition, distances, profile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Three modalities of spatial omics

    Spatial datasets combine complementary views of the same cells. This notebook walks through each modality, then uses **LandmarksWidget** to measure local **proximity**, **composition**, and **gradients** on the tissue map.

    | Modality | What it captures | Representation here |
    | --- | --- | --- |
    | **Spatial** | Where cells sit in tissue | `x`, `y` coordinates |
    | **RNA** | Transcriptional state | UMAP of expression |
    | **Morphology** | Nucleus shape / size | Shapely geometry features |

    Data: mouse liver SpatialData store (`data/mouse_liver.zarr`).
    """)
    return


@app.cell
def _(sd):
    sdata = sd.read_zarr("data/mouse_liver.zarr")
    table = sdata.tables["table"]
    nuclei = sdata.shapes["nucleus_boundaries"]
    return nuclei, table


@app.cell
def _(np, nuclei, pd, table):
    xy_df = pd.DataFrame(
        table.obsm["spatial"],
        columns=["x", "y"],
        index=table.obs["cell_ID"].astype(str),
    )
    xy_df["cell_type"] = table.obs["annotation"].to_numpy()

    # Align nucleus polygons to table cell IDs
    _gdf = nuclei.copy()
    _gdf.index = _gdf.index.astype(str)
    _gdf = _gdf.loc[xy_df.index]

    def _axis_lengths(geom):
        rect = geom.minimum_rotated_rectangle
        if rect.is_empty or geom.is_empty:
            return 0.0, 0.0
        coords = list(rect.exterior.coords)
        edges = [
            float(np.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]))
            for i in range(4)
        ]
        return max(edges), min(edges)

    _maj, _min = zip(*[_axis_lengths(g) for g in _gdf.geometry], strict=True)
    _maj = np.asarray(_maj, dtype=float)
    _min = np.asarray(_min, dtype=float)
    _area = _gdf.geometry.area.to_numpy(dtype=float)
    _convex = _gdf.geometry.convex_hull.area.to_numpy(dtype=float)
    _perim = _gdf.geometry.length.to_numpy(dtype=float)

    morph_raw = pd.DataFrame(
        {
            "area": _area,
            "area_convex": _convex,
            "axis_major_length": _maj,
            "axis_minor_length": _min,
            "eccentricity": np.sqrt(
                np.clip(1.0 - np.where(_maj > 0, (_min / _maj) ** 2, 1.0), 0.0, 1.0)
            ),
            "perimeter": _perim,
            "solidity": np.where(_convex > 0, _area / _convex, np.nan),
        },
        index=xy_df.index,
    )
    return morph_raw, xy_df


@app.cell
def _(StandardScaler, morph_raw, pd, xy_df):
    _cols = list(morph_raw.columns)
    morph_df = pd.DataFrame(
        StandardScaler().fit_transform(morph_raw[_cols]),
        index=morph_raw.index,
        columns=_cols,
    )
    morph_df["cell_type"] = xy_df["cell_type"].to_numpy()
    return (morph_df,)


@app.cell
def _(sc, table):
    def _transform(adata):
        out = adata.copy()
        sc.pp.log1p(out)
        sc.pp.pca(out)
        sc.pp.neighbors(out)
        sc.tl.umap(out)
        return out

    rna_adata = _transform(table)
    return (rna_adata,)


@app.cell
def _(pd, rna_adata, xy_df):
    rna_df = pd.DataFrame(
        rna_adata.obsm["X_umap"],
        index=xy_df.index,
        columns=["umap_1", "umap_2"],
    )
    rna_df["cell_type"] = xy_df["cell_type"].to_numpy()
    return (rna_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Spatial modality

    Tissue coordinates answer *where* a cell is. Neighborhood structure — who sits next to whom — is the substrate for proximity and composition measurements below.
    """)
    return


@app.cell
def _(alt, mo, xy_df):
    xy_chart = mo.ui.altair_chart(
        alt.Chart(xy_df.reset_index())
        .mark_point(filled=True, opacity=0.75, size=18)
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("y:Q", title="y"),
            color=alt.Color("cell_type:N", legend=alt.Legend(title="cell type", columns=2)),
            tooltip=["cell_ID:N", "cell_type:N"],
        )
        .properties(height=360, title="Tissue map")
    )
    xy_chart
    return (xy_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. RNA modality

    Expression lives in high dimension. A UMAP summarizes transcriptional neighborhoods: cells that co-cluster here often share programs, even when they are far apart on the tissue map.
    """)
    return


@app.cell
def _(alt, mo, rna_df):
    rna_chart = mo.ui.altair_chart(
        alt.Chart(rna_df.reset_index())
        .mark_point(filled=True, opacity=0.75, size=18)
        .encode(
            x=alt.X("umap_1:Q", title="UMAP 1"),
            y=alt.Y("umap_2:Q", title="UMAP 2"),
            color=alt.Color("cell_type:N", legend=None),
            tooltip=["cell_ID:N", "cell_type:N"],
        )
        .properties(height=360, title="RNA UMAP")
    )
    rna_chart
    return (rna_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Morphology modality

    Nucleus polygons already live as **GeoPandas** geometries. Shape descriptors (area, perimeter, eccentricity, solidity, …) are computed with **Shapely** — no image rasterization and no Squidpy feature extraction.

    Parallel coordinates show how these descriptors vary by cell type after z-scoring.
    """)
    return


@app.cell
def _(ParallelCoordinates, mo, morph_df):
    morph_chart = ParallelCoordinates(morph_df, color_by="cell_type")
    mo.vstack(
        [
            mo.md("_Morphology features from `nucleus_boundaries` via shapely/geopandas._"),
            morph_chart,
        ]
    )
    return


@app.cell
def _(mo, rna_chart, xy_chart):
    mo.hstack([xy_chart, rna_chart], widths="equal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Landmarks: measure tissue structure

    1. Place **landmarks** (point / line / spline / shape).
    2. Optionally draw a **selection** and keep it selected — measurements use that
       region only. Deselect / clear selections ⇒ all cells.
    3. Explore **Distance** (box+strip), **Composition** (shapes), **Profile** (line/spline).
    """)
    return


@app.cell
def _(LandmarksWidget, plt, xy_df):
    _fig, _ax = plt.subplots(figsize=(6.5, 6.5))
    _cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    group_colors = {}
    for _i, (_ct, _chunk) in enumerate(xy_df.groupby("cell_type")):
        _color = _cycle[_i % len(_cycle)]
        group_colors[_ct] = _color
        _ax.scatter(
            _chunk["x"], _chunk["y"], s=6, alpha=0.55, label=_ct, color=_color
        )
    _ax.set_aspect("equal")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title("Landmarks then selections")
    _ax.legend(fontsize=7, loc="upper right", frameon=False, markerscale=2)
    landmarks = LandmarksWidget(_fig, mode="select")
    plt.close(_fig)
    return group_colors, landmarks


@app.cell
def _(landmarks, mo):
    landmarks_ui = mo.ui.anywidget(landmarks)
    landmarks_ui
    return (landmarks_ui,)


@app.cell
def _(landmarks_ui, mo):
    _lm_ids = [str(lm.get("id")) for lm in landmarks_ui.landmarks if not lm.get("hidden")]
    _sel_ids = ["all"] + [str(s.get("id")) for s in landmarks_ui.selections]
    landmark_pick = mo.ui.dropdown(
        options=_lm_ids or ["(none)"],
        value=_lm_ids[0] if _lm_ids else "(none)",
        label="Landmark",
    )
    selection_pick = mo.ui.dropdown(
        options=_sel_ids,
        value="all",
        label="Selection",
    )
    plot_type = mo.ui.radio(
        options=["Distance", "Composition", "Profile"],
        value="Distance",
        label="Plot",
    )
    mo.hstack([landmark_pick, selection_pick, plot_type], gap=1)
    return landmark_pick, plot_type, selection_pick


@app.cell
def _(
    composition,
    distances,
    group_colors,
    landmark_pick,
    landmarks_ui,
    matplotlib,
    mo,
    plot_type,
    plt,
    profile,
    selection_pick,
    sns,
    theme,
    xy_df,
):
    matplotlib.style.use("dark_background" if theme == "dark" else "default")
    _x = xy_df["x"].to_numpy()
    _y = xy_df["y"].to_numpy()
    _groups = xy_df["cell_type"].to_numpy()
    _sid = selection_pick.value
    _idx = landmarks_ui.get_indices(_x, _y, selection_id=_sid)
    _hue_order = list(group_colors.keys())
    _lid = landmark_pick.value
    _lms = [lm for lm in landmarks_ui.landmarks if str(lm.get("id")) == _lid]

    if not _lms:
        chart = mo.md("_Add a landmark to explore measurements._")
    elif plot_type.value == "Distance":
        _d = distances(_x, _y, _groups, _lms, _idx)
        if _d.empty:
            chart = mo.md("_No distance rows._")
        else:
            _fig, _ax = plt.subplots(figsize=(6, 3.8))
            sns.boxplot(
                data=_d,
                x="group",
                y="distance",
                order=_hue_order,
                hue="group",
                hue_order=_hue_order,
                palette=group_colors,
                ax=_ax,
                legend=False,
            )
            sns.stripplot(
                data=_d,
                x="group",
                y="distance",
                order=_hue_order,
                hue="group",
                hue_order=_hue_order,
                palette=group_colors,
                ax=_ax,
                size=2.5,
                alpha=0.3,
                legend=False,
            )
            _ax.set_title(f"Distance · {_lid} · selection={_sid}")
            _fig.tight_layout()
            chart = _fig
    elif plot_type.value == "Composition":
        _c = composition(_x, _y, _groups, _lms, _idx)
        if _c.empty:
            chart = mo.md("_No shape ∩ selection points (need a **shape** landmark)._")
        else:
            _fig, _ax = plt.subplots(figsize=(6, 3.8))
            sns.barplot(
                data=_c,
                x="group",
                y="proportion",
                order=_hue_order,
                hue="group",
                hue_order=_hue_order,
                palette=group_colors,
                ax=_ax,
                legend=False,
            )
            _ax.set_title(f"Composition · {_lid} · selection={_sid}")
            _fig.tight_layout()
            chart = _fig
    else:
        _p = profile(_x, _y, _groups, _lms, _idx, n=30)
        if _p.empty:
            chart = mo.md("_No profile rows (need **line** / **spline** landmarks)._")
        else:
            _fig, _ax = plt.subplots(figsize=(6, 3.8))
            sns.lineplot(
                data=_p,
                x="s",
                y="proportion",
                hue="group",
                hue_order=_hue_order,
                palette=group_colors,
                ax=_ax,
            )
            _ax.set_xlabel("along path (s)")
            _ax.set_title(f"Profile · {_lid} · selection={_sid}")
            _fig.tight_layout()
            chart = _fig
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the measurements

    The widget outputs **landmarks** and **selections**. Notebook dropdowns
    pick which landmark/selection to measure via ``get_indices(..., selection_id=…)``.

    - **Distance** — per-cell distance to each landmark, grouped by cell type.
    - **Composition** — cell-type mix inside each **shape**.
    - **Profile** — local mix along each **line/spline**.

    Cross-check against the RNA UMAP and morphology parallel coordinates.
    """)
    return


if __name__ == "__main__":
    app.run()
