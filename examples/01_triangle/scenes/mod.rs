mod common;
mod triangle;

pub use triangle::Triangle;

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

pub trait SceneState {
    fn handle_input(&mut self, input: SceneInput);
    fn update(&mut self, frame: FrameInput) -> Result<(), String>;
    fn record(&mut self, context: u64) -> Result<(), String>;
    fn destroy(self: Box<Self>, context: u64);
}
