use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct StacktraceEntry {
    pub process_name: String,
    pub start_ms: u128,
    pub end_ms: u128,
}

static REGISTRY: Mutex<Vec<StacktraceEntry>> = Mutex::new(Vec::new());
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn init() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub fn register(name: &str, start_ms: u128, end_ms: u128) {
    if let Ok(mut entries) = REGISTRY.lock() {
        entries.push(StacktraceEntry {
            process_name: name.to_string(),
            start_ms,
            end_ms,
        });
    }
}

pub fn flush_csv(path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let entries = REGISTRY.lock().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::Other, "lock poisoned")
    })?;

    let base = entries.iter().map(|e| e.start_ms).min().unwrap_or(0);

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "process_name,start_ms,end_ms")?;
    for entry in entries.iter() {
        let start = entry.start_ms.saturating_sub(base);
        let end = entry.end_ms.saturating_sub(base);
        writeln!(file, "{},{},{}", entry.process_name, start, end)?;
    }
    file.flush()
}

#[macro_export]
macro_rules! stacktrace {
    ($id:expr, $block:expr) => {{
        if $crate::shared::stacktrace::is_enabled() {
            let __start = $crate::shared::stacktrace::now_ms();
            let __result = $block;
            $crate::shared::stacktrace::register($id, __start, $crate::shared::stacktrace::now_ms());
            __result
        } else {
            $block
        }
    }};
}
