use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameInput {
    pub width: u32,
    pub height: u32,
    pub delta_seconds: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneKey {
    Tab,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Insert,
    Delete,
    Backspace,
    Space,
    Enter,
    Escape,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneInput {
    CursorMoved { x: f64, y: f64 },
    PrimaryButton(bool),
    ScrollLines(f32),
    Character(char),
    Key { key: SceneKey, pressed: bool },
}

pub fn dispatch_window_input(event: &WindowEvent, mut dispatch: impl FnMut(SceneInput)) -> bool {
    match event {
        WindowEvent::CursorMoved { position, .. } => dispatch(SceneInput::CursorMoved {
            x: position.x,
            y: position.y,
        }),
        WindowEvent::MouseInput {
            state,
            button: MouseButton::Left,
            ..
        } => {
            dispatch(SceneInput::PrimaryButton(*state == ElementState::Pressed));
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let lines = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(position) => position.y as f32 / 24.0,
            };
            dispatch(SceneInput::ScrollLines(lines));
        }
        WindowEvent::KeyboardInput { event, .. } => {
            let pressed = event.state == ElementState::Pressed;
            dispatch(SceneInput::Key {
                key: scene_key(&event.logical_key),
                pressed,
            });
            if pressed && let Key::Character(text) = &event.logical_key {
                for character in text.chars() {
                    dispatch(SceneInput::Character(character));
                }
            }
        }
        _ => return false,
    }
    true
}

fn scene_key(key: &Key) -> SceneKey {
    match key {
        Key::Named(NamedKey::Tab) => SceneKey::Tab,
        Key::Named(NamedKey::ArrowLeft) => SceneKey::Left,
        Key::Named(NamedKey::ArrowRight) => SceneKey::Right,
        Key::Named(NamedKey::ArrowUp) => SceneKey::Up,
        Key::Named(NamedKey::ArrowDown) => SceneKey::Down,
        Key::Named(NamedKey::PageUp) => SceneKey::PageUp,
        Key::Named(NamedKey::PageDown) => SceneKey::PageDown,
        Key::Named(NamedKey::Home) => SceneKey::Home,
        Key::Named(NamedKey::End) => SceneKey::End,
        Key::Named(NamedKey::Insert) => SceneKey::Insert,
        Key::Named(NamedKey::Delete) => SceneKey::Delete,
        Key::Named(NamedKey::Backspace) => SceneKey::Backspace,
        Key::Named(NamedKey::Space) => SceneKey::Space,
        Key::Named(NamedKey::Enter) => SceneKey::Enter,
        Key::Named(NamedKey::Escape) => SceneKey::Escape,
        _ => SceneKey::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping_covers_named_and_unknown_keys() {
        assert_eq!(scene_key(&Key::Named(NamedKey::Escape)), SceneKey::Escape);
        assert_eq!(scene_key(&Key::Character("x".into())), SceneKey::Other);
    }
}
