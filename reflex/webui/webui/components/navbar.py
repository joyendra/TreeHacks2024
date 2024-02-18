import reflex as rx

from webui import styles
from webui.state import State


def navbar():
    return rx.chakra.box(
        rx.chakra.hstack(
            rx.chakra.hstack(
                rx.chakra.button(
                    "+ New chat",
                    bg=styles.accent_color,
                    px="4",
                    py="2",
                    h="auto",
                    on_click=State.toggle_modal,
                ),
                spacing="8",
            ),
            justify="space-between",
        ),
        bg=styles.bg_dark_color,
        backdrop_filter="auto",
        backdrop_blur="lg",
        p="4",
        border_bottom=f"1px solid {styles.border_color}",
        position="sticky",
        top="0",
        z_index="100",
    )
