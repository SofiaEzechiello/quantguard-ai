"""funções utilitárias para gráficos"""

def configurar_figura(fig, y_title=None, x_title=None):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", color="#626272"),
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        gridcolor="#EEEEF3",
        zerolinecolor="#EEEEF3",
        title=x_title,
    )

    fig.update_yaxes(
        gridcolor="#EEEEF3",
        zerolinecolor="#EEEEF3",
        title=y_title,
    )

    return fig


