import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import spatialdata as sd
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import seaborn as sns
    from wigglystuff import LandmarksWidget

    return LandmarksWidget, gpd, matplotlib, mo, np, pd, plt, sd, sns


@app.cell
def _(matplotlib, mo):
    theme = mo.app_meta().theme
    matplotlib.style.use("dark_background" if theme == "dark" else "default")
    return (theme,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LandmarksWidget: draw → measure

    Mouse liver SpatialData (`data/mouse_liver.zarr`). The reactive loop:

    1. **Draw a landmark** (point / line / spline / shape) on the tissue map
    2. Optionally **restrict** with a selection (lasso / polygon / …)
    3. Measure **relationships** between that landmark and cells
    """)
    return


@app.cell
def _(sd):
    sdata = sd.read_zarr("data/mouse_liver.zarr")
    table = sdata.tables["table"]
    return (table,)


@app.cell
def _(pd, table):
    xy_df = pd.DataFrame(
        table.obsm["spatial"],
        columns=["x", "y"],
        index=table.obs["cell_ID"].astype(str),
    )
    xy_df["cell_type"] = table.obs["annotation"].to_numpy()
    return (xy_df,)


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


    def _row_density(df):
        """Max-normalize counts within each landmark × cell type (peak = 1)."""
        if df.empty:
            out = df.copy()
            out["density"] = pd.Series(dtype=float)
            return out
        out = df.copy()
        maxima = out.groupby(["landmark_id", "group"], sort=False)["count"].transform("max")
        out["density"] = np.where(maxima > 0, out["count"] / maxima, np.nan)
        return out


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
        """Cell-type density along a line/spline (for Gradient-along heatmaps)."""
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
                n_pts = int(neighborhood.sum())
                freq = {}
                if n_pts:
                    values, counts = np.unique(local_groups, return_counts=True)
                    freq = {v: int(c) for v, c in zip(values, counts, strict=True)}
                for g in all_groups:
                    rows.append(
                        {
                            "landmark_id": lid,
                            "landmark_type": ltype,
                            "s": s,
                            "s_label": f"{s:.2f}",
                            "x": float(pt.x),
                            "y": float(pt.y),
                            "n_points": n_pts,
                            "group": g,
                            "count": int(freq.get(g, 0)),
                        }
                    )
        return _row_density(pd.DataFrame(rows))


    def _nice_ceil(value):
        """Ceil to the next 1/2/5 × 10^k boundary."""
        value = float(value)
        if value <= 0:
            return 1.0
        exp = int(np.floor(np.log10(value)))
        base = 10.0 ** exp
        for m in (1, 2, 5, 10):
            if m * base >= value:
                return float(m * base)
        return float(10 * base)


    def _nice_step(limit, target_ticks=4):
        """Pick a 1/2/5 × 10^k tick step for ~target_ticks marks from 0..limit."""
        limit = float(limit)
        if limit <= 0:
            return 1.0
        raw = limit / max(target_ticks - 1, 1)
        exp = int(np.floor(np.log10(raw))) if raw > 0 else 0
        base = 10.0 ** exp
        for m in (1, 2, 5, 10):
            if m * base >= raw:
                return float(m * base)
        return float(10 * base)


    def distance_bands(x, y, groups, landmarks, indices, edges=None, n_bins=16):
        """Cell-type density in distance bins away from a landmark (Gradient-perp)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        indices = np.asarray(indices, dtype=int)
        if edges is None:
            span = max(float(np.ptp(x) or 1.0), float(np.ptp(y) or 1.0))
            hi = _nice_ceil(0.12 * span)
            edges = np.linspace(0.0, hi, n_bins + 1).tolist()
        points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        cand = np.zeros(len(x), dtype=bool)
        cand[indices] = True
        all_groups = [g for g in pd.unique(groups) if g is not None]
        rows = []
        for lid, (ltype, geom) in landmark_geoms(landmarks).items():
            dist = points.distance(geom).to_numpy()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                band = cand & (dist >= lo) & (dist < hi)
                n = int(band.sum())
                mid = 0.5 * (lo + hi)
                label = f"{mid:g}"
                freq = {}
                if n:
                    subset = groups[band]
                    values, counts = np.unique(subset, return_counts=True)
                    freq = {v: int(c) for v, c in zip(values, counts, strict=True)}
                for g in all_groups:
                    rows.append(
                        {
                            "landmark_id": lid,
                            "landmark_type": ltype,
                            "band": label,
                            "band_lo": lo,
                            "band_hi": hi,
                            "band_mid": mid,
                            "group": g,
                            "count": int(freq.get(g, 0)),
                            "n_total": n,
                        }
                    )
        return _row_density(pd.DataFrame(rows))


    return composition, distances, landmark_geoms


@app.cell
def _(LandmarksWidget, plt, xy_df):
    _fig, _ax = plt.subplots(figsize=(10,7))
    _cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    group_colors = {}
    # Draw abundant types first so rare niche types stay visible on top
    _order = xy_df["cell_type"].value_counts().index.tolist()
    for _i, _ct in enumerate(_order):
        _chunk = xy_df[xy_df["cell_type"] == _ct]
        _color = _cycle[_i % len(_cycle)]
        group_colors[_ct] = _color
        _ax.scatter(
            _chunk["x"],
            _chunk["y"],
            s=7 if _ct == "Hepatocytes" else 14,
            alpha=0.45 if _ct == "Hepatocytes" else 0.75,
            label=_ct,
            color=_color,
            zorder=1 if _ct == "Hepatocytes" else 2,
        )
    _ax.set_aspect("equal")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title("Mouse liver · draw landmarks, then measure")
    _ax.legend(
        bbox_to_anchor=(1,0.8),
        frameon=False,
        markerscale=3,
    )
    landmarks = LandmarksWidget(_fig, mode="point")
    plt.close(_fig)
    return group_colors, landmarks


@app.cell
def _(landmarks, mo):
    landmarks_ui = mo.ui.anywidget(landmarks)
    landmarks_ui
    return (landmarks_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Instructions**

    Choose a use case or customize parameters yourself.

    **Draw** a landmark on the tissue map (point / line / spline / shape).
    Optionally add a selection (lasso / polygon / …) to restrict which cells are measured.

    **Measure** with Distance / Gradient (perpendicular) / Gradient (along) / Composition —
    or let a premade use case select the relationship for you.

    | Use case | Draw | Measure |
    | --- | --- | --- |
    | Lobule zonation | points on portal / central veins | Distance |
    | Perivascular belt | line along a vessel | Gradient (perpendicular) |
    | Porto-central axis | spline portal → central | Gradient (along) |
    | Niche composition | closed shape around a niche | Composition |
    | ROI-restricted | landmark + selection ROI | any, masked |
    | Immune proximity | point on a niche | Distance |
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    use_case = mo.ui.radio(
        options=[
            "1. Lobule zonation",
            "2. Perivascular belt",
            "3. Porto-central axis",
            "4. Niche composition",
            "5. ROI-restricted",
            "6. Immune proximity",
        ],
        value="1. Lobule zonation",
        label="",
    )
    mo.vstack(
        [
            mo.md(
                "**Premade Use Cases**\n\n"
                "These use-cases simply select the proper relationship to visualize."
            ),
            use_case,
        ],
        gap=0.4,
    )
    return (use_case,)


@app.cell
def _(landmarks_ui, mo, use_case):
    _lm_ids = [str(lm.get("id")) for lm in landmarks_ui.landmarks if not lm.get("hidden")]
    _sel_ids = ["all"] + [str(s.get("id")) for s in landmarks_ui.selections]
    _default_plot = {
        "1. Lobule zonation": "Distance",
        "2. Perivascular belt": "Gradient (perpendicular)",
        "3. Porto-central axis": "Gradient (along)",
        "4. Niche composition": "Composition",
        "5. ROI-restricted": "Distance",
        "6. Immune proximity": "Distance",
    }[use_case.value]
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
        options=[
            "Distance",
            "Gradient (perpendicular)",
            "Composition",
            "Gradient (along)",
        ],
        value=_default_plot,
        label="Relationship",
    )
    mo.hstack([landmark_pick, selection_pick, plot_type], gap=1)
    return landmark_pick, plot_type, selection_pick


@app.cell
def _(
    composition,
    distances,
    gpd,
    group_colors,
    landmark_geoms,
    landmark_pick,
    landmarks_ui,
    matplotlib,
    mo,
    np,
    pd,
    plot_type,
    plt,
    selection_pick,
    sns,
    theme,
    use_case,
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

    _immune_focus = {
        "Hepatocytes",
        "Kupffer cells",
        "B cells",
        "Other immunecells",
        "Cholangiocytes",
        "stellate",
        "LSEC Portal",
        "LSEC Central",
    }

    _grad_focus = [
        "Kupffer cells",
        "stellate",
        "LSEC Portal",
        "LSEC Central",
        "B cells",
        "Cholangiocytes",
        "Fibroblast",
        "Hepatocytes",
    ]


    def _nice_limit_and_step(limit, target_ticks=4):
        limit = float(max(limit, 0.0))
        if limit <= 0:
            return 1.0, 1.0
        exp = int(np.floor(np.log10(limit)))
        base = 10.0 ** exp
        nice_hi = next((m * base for m in (1, 2, 5, 10) if m * base >= limit), 10 * base)
        raw = nice_hi / max(target_ticks - 1, 1)
        exp = int(np.floor(np.log10(raw))) if raw > 0 else 0
        base = 10.0 ** exp
        step = next((m * base for m in (1, 2, 5, 10) if m * base >= raw), 10 * base)
        return nice_hi, step


    def _hist_heatmap(
        df,
        value_col,
        row_order,
        title,
        xlabel,
        x_max=None,
        x_min=0.0,
        n_bins=24,
    ):
        """Max-normalized row heatmap rendered by seaborn.histplot."""
        rows = []
        for group in row_order:
            values = df.loc[df["group"] == group, value_col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                rows.append((group, values))
        if not rows:
            return None

        x_min = float(x_min)
        data_max = max(float(values.max()) for _, values in rows)
        if x_max is None:
            x_max, step = _nice_limit_and_step(data_max)
        else:
            x_max = float(x_max)
            _, step = _nice_limit_and_step(max(x_max - x_min, 1e-9))
        x_edges = np.linspace(x_min, x_max, n_bins + 1)
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])

        weighted = []
        shown_groups = []
        for group, values in rows:
            counts, _ = np.histogram(values, bins=x_edges)
            peak = int(counts.max())
            if peak <= 0:
                continue
            row = len(shown_groups)
            shown_groups.append(group)
            for mid, count in zip(x_mid, counts, strict=True):
                if count:
                    weighted.append(
                        {
                            "x": float(mid),
                            "row": row,
                            "density": float(count) / peak,
                        }
                    )
        if not weighted:
            return None

        plot_df = pd.DataFrame(weighted)
        y_edges = np.arange(-0.5, len(shown_groups) + 0.5, 1.0)
        edge = "#f1f5f9" if theme == "dark" else "#0f172a"
        grid = "#94a3b8" if theme == "dark" else "#64748b"
        fig, ax = plt.subplots(
            figsize=(7.2, max(2.8, 0.38 * len(shown_groups) + 1.2))
        )
        sns.histplot(
            data=plot_df,
            x="x",
            y="row",
            weights="density",
            bins=(x_edges, y_edges),
            cmap="Reds",
            vmin=0,
            vmax=1,
            cbar=True,
            cbar_kws={"label": "Density (Norm.)", "ticks": [0, 0.5, 1], "shrink": 0.8},
            ax=ax,
        )

        ticks = np.arange(x_min, x_max + 0.5 * step, step)
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", length=5, width=1.0, color=edge, direction="out")
        for tick in ticks:
            ax.axvline(tick, color=grid, linestyle=":", linewidth=0.8, zorder=3)
        ax.set_yticks(np.arange(len(shown_groups)))
        ax.set_yticklabels(shown_groups)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(edge)
            spine.set_linewidth(1.1)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cell type")
        ax.set_title(title)
        fig.tight_layout()
        return fig


    def _along_positions(x, y, groups, landmarks, indices, radius=None):
        """Project selected cells onto line/spline landmarks → normalized path position s."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        indices = np.asarray(indices, dtype=int)
        points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        if radius is None:
            radius = 0.05 * max(float(np.ptp(x) or 1.0), float(np.ptp(y) or 1.0))
        rows = []
        for lid, (ltype, geom) in landmark_geoms(landmarks).items():
            if ltype not in ("line", "spline") or geom.geom_type != "LineString":
                continue
            dist = points.distance(geom).to_numpy()
            for i in indices:
                if dist[i] > radius:
                    continue
                s = float(geom.project(points.iloc[i], normalized=True))
                rows.append(
                    {
                        "landmark_id": lid,
                        "landmark_type": ltype,
                        "group": groups[i],
                        "s": s,
                        "distance": float(dist[i]),
                    }
                )
        return pd.DataFrame(rows)


    if not _lms:
        chart = mo.md("_Add a landmark on the map to unlock measurements._")
    elif plot_type.value == "Distance":
        _d = distances(_x, _y, _groups, _lms, _idx)
        if use_case.value == "6. Immune proximity" and not _d.empty:
            _d = _d[_d["group"].isin(_immune_focus)]
            _order = [g for g in _hue_order if g in set(_d["group"])]
        else:
            _order = _hue_order
        if _d.empty:
            chart = mo.md("_No distance rows for this landmark / selection._")
        else:
            _fig, _ax = plt.subplots(figsize=(7.2, 3.8))
            sns.boxplot(
                data=_d,
                x="group",
                y="distance",
                order=_order,
                hue="group",
                hue_order=_order,
                palette=group_colors,
                ax=_ax,
                legend=False,
            )
            sns.stripplot(
                data=_d,
                x="group",
                y="distance",
                order=_order,
                hue="group",
                hue_order=_order,
                palette=group_colors,
                ax=_ax,
                size=2.2,
                alpha=0.28,
                legend=False,
            )
            _ax.tick_params(axis="x", rotation=35)
            _ax.set_title(f"Distance · {_lid} · selection={_sid}")
            _fig.tight_layout()
            chart = _fig
    elif plot_type.value == "Gradient (perpendicular)":
        _d = distances(_x, _y, _groups, _lms, _idx)
        if _d.empty:
            chart = mo.md("_No distance rows for this landmark / selection._")
        else:
            _focus = [g for g in _grad_focus if g in set(_d["group"])]
            _hi, _ = _nice_limit_and_step(float(_d["distance"].quantile(0.98)))
            chart = _hist_heatmap(
                _d,
                "distance",
                _focus,
                f"Gradient (perpendicular) · {_lid} · selection={_sid}",
                "distance from landmark",
                x_max=_hi,
            ) or mo.md("_No cells in distance bins for this landmark / selection._")
    elif plot_type.value == "Composition":
        _c = composition(_x, _y, _groups, _lms, _idx)
        if _c.empty:
            chart = mo.md("_Need a **shape** landmark that covers cells._")
        else:
            _fig, _ax = plt.subplots(figsize=(7.2, 3.8))
            sns.barplot(
                data=_c,
                x="group",
                y="proportion",
                order=[g for g in _hue_order if g in set(_c["group"])],
                hue="group",
                hue_order=_hue_order,
                palette=group_colors,
                ax=_ax,
                legend=False,
            )
            _ax.tick_params(axis="x", rotation=35)
            _n = int(_c["n_total"].iloc[0]) if len(_c) else 0
            _ax.set_title(f"Composition · {_lid} · n={_n} · selection={_sid}")
            _fig.tight_layout()
            chart = _fig
    else:
        _p = _along_positions(_x, _y, _groups, _lms, _idx)
        if _p.empty:
            chart = mo.md("_Need a **line** or **spline** landmark for Gradient (along)._")
        else:
            _focus = [g for g in _grad_focus if g in set(_p["group"])]
            chart = _hist_heatmap(
                _p,
                "s",
                _focus,
                f"Gradient (along) · {_lid} · selection={_sid}",
                "along path (start → end)",
                x_min=0.0,
                x_max=1.0,
            ) or mo.md("_No nearby cells in path bins._")

    _n_sel = int(len(_idx))
    _n_all = int(len(_x))
    summary = mo.md(
        f"_Selection `{_sid}` → **{_n_sel} / {_n_all}** cells · "
        f"use case: **{use_case.value}**_"
    )
    mo.vstack([summary, chart])

    return


if __name__ == "__main__":
    app.run()
