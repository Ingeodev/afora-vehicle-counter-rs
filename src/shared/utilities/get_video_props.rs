use ffmpeg_next as ffmpeg;
use std::path::Path;


#[derive(Debug)]
pub struct VideoProperties {
    pub fps: u32,
    pub width: u32,
    pub height: u32,
}



pub fn get_video_properties<P: AsRef<Path>>(path: P) -> Result<VideoProperties, ffmpeg::Error> {
    // Es crucial que ffmpeg::init() se haya ejecutado al menos una vez en tu aplicación
    // antes de llamar a esta función. Puedes descomentar la siguiente línea si prefieres
    // inicializarlo aquí (es seguro llamarlo múltiples veces).
    // ffmpeg::init()?;

    // 1. Abrir el archivo de video
    let ictx = ffmpeg::format::input(&path)?;

    // 2. Buscar el mejor stream que sea de tipo Video
    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;

    // 3. Obtener el contexto del decodificador para leer width y height
    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    let decoder = context.decoder().video()?;

    let width = decoder.width();
    let height = decoder.height();

    // 4. Calcular los FPS
    // FFmpeg almacena los FPS como un número racional (fracción: numerador / denominador)
    let frame_rate = stream.avg_frame_rate();

    let fps = if frame_rate.denominator() > 0 {
        // Obtenemos el valor real en f64 primero
        let exact_fps = f64::from(frame_rate.numerator()) / f64::from(frame_rate.denominator());
        // Lo redondeamos al entero más cercano y lo casteamos a u32
        exact_fps.round() as u32
    } else {
        0 // Valor por defecto si falla la lectura
    };

    Ok(VideoProperties { fps, width, height })
}