#!/usr/bin/env python3
"""
Shared MATLAB-style plotting configuration for paper-quality figures.

Usage:
    from examples.organized.plot_style import apply_matlab_style, COLORS, MARKERS, save_fig
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt


# ── Protocol visual identity ────────────────────────────────────────────────
COLORS = {
    'DR8':      '#0072BD',   # MATLAB blue
    'DR9':      '#D95319',   # MATLAB orange
    'Semantic': '#77AC30',   # MATLAB green
}

MARKERS = {
    'DR8':      'o',
    'DR9':      's',
    'Semantic': '^',
}

LINE_STYLES = {
    'DR8':      '-',
    'DR9':      '--',
    'Semantic': '-.',
}


def apply_matlab_style() -> None:
    """Apply a MATLAB-like visual style globally to matplotlib."""
    matplotlib.rcParams.update({
        # ── Font ──────────────────────────────────────────────────────
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset':  'stix',
        'font.size':         11,

        # ── Axes ──────────────────────────────────────────────────────
        'axes.linewidth':    1.0,
        'axes.grid':         True,
        'axes.titlesize':    12,
        'axes.labelsize':    12,
        'axes.labelweight':  'normal',
        'axes.facecolor':    'white',
        'axes.edgecolor':    'black',
        'axes.prop_cycle':   matplotlib.cycler(color=[
            '#0072BD', '#D95319', '#77AC30', '#7E2F8E',
            '#EDB120', '#4DBEEE', '#A2142F',
        ]),

        # ── Grid ──────────────────────────────────────────────────────
        'grid.color':        '#CCCCCC',
        'grid.linestyle':    '--',
        'grid.linewidth':    0.5,
        'grid.alpha':        0.7,

        # ── Lines ─────────────────────────────────────────────────────
        'lines.linewidth':   1.5,
        'lines.markersize':  7,

        # ── Ticks ─────────────────────────────────────────────────────
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.major.size':  5,
        'ytick.major.size':  5,
        'xtick.minor.size':  3,
        'ytick.minor.size':  3,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'xtick.top':         True,
        'ytick.right':       True,

        # ── Legend ────────────────────────────────────────────────────
        'legend.frameon':    True,
        'legend.framealpha': 1.0,
        'legend.edgecolor':  'black',
        'legend.fontsize':   10,
        'legend.fancybox':   False,

        # ── Figure ────────────────────────────────────────────────────
        'figure.facecolor':  'white',
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.05,
    })


def save_fig(fig: plt.Figure, output_dir: str, basename: str) -> None:
    """Save figure as PNG, EPS and JPG (300 DPI, tight)."""
    for ext in ('png', 'eps', 'jpg'):
        path = os.path.join(output_dir, f'{basename}.{ext}')
        fmt = 'jpeg' if ext == 'jpg' else ext
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        print(f'[OK] {path}')
    plt.close(fig)


def protocol_plot_kwargs(protocol: str) -> dict:
    """Return common plot kwargs for a given protocol."""
    return {
        'color':          COLORS[protocol],
        'marker':         MARKERS[protocol],
        'linestyle':      LINE_STYLES[protocol],
        'linewidth':      1.5,
        'markerfacecolor': 'white',
        'markeredgewidth': 1.4,
        'markeredgecolor': COLORS[protocol],
        'markersize':     7,
        'label':          protocol,
    }
