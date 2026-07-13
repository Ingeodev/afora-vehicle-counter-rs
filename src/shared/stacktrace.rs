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
static ACTIVE_TAGS: Mutex<Option<Vec<String>>> = Mutex::new(None);

pub fn init(tags: Option<&str>) {
    ENABLED.store(true, Ordering::Relaxed);
    if let Some(raw) = tags {
        let parsed: Vec<String> = raw
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if !parsed.is_empty() {
            *ACTIVE_TAGS.lock().unwrap() = Some(parsed);
        }
    }
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn should_record(entry_tags: &str) -> bool {
    if !is_enabled() {
        return false;
    }
    let active = ACTIVE_TAGS.lock().unwrap();
    match active.as_ref() {
        None => true,
        Some(filter_tags) => {
            if entry_tags.is_empty() {
                return false;
            }
            entry_tags
                .split(',')
                .map(|t| t.trim())
                .any(|t| filter_tags.iter().any(|f| f == t))
        }
    }
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
    use std::collections::HashMap;

    let entries = REGISTRY.lock().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::Other, "lock poisoned")
    })?;

    let base = entries.iter().map(|e| e.start_ms).min().unwrap_or(0);

    let mut counters: HashMap<&str, usize> = HashMap::new();
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "Paso\tInicio\tFin\tDuración")?;
    for entry in entries.iter() {
        let counter = counters.entry(&entry.process_name).or_insert(0);
        *counter += 1;
        let start = entry.start_ms.saturating_sub(base);
        let end = entry.end_ms.saturating_sub(base);
        let duration = end.saturating_sub(start);
        writeln!(file, "{} {}\t{}\t{}\t{}", entry.process_name, counter, start, end, duration)?;
    }
    file.flush()
}

#[macro_export]
macro_rules! stacktrace {
    ($id:expr, $tags:expr, $block:expr) => {{
        if $crate::shared::stacktrace::should_record($tags) {
            let __start = $crate::shared::stacktrace::now_ms();
            let __result = $block;
            $crate::shared::stacktrace::register($id, __start, $crate::shared::stacktrace::now_ms());
            __result
        } else {
            $block
        }
    }};
}
