use std::sync::Arc;
use std::thread::JoinHandle;
use flume::{unbounded, Sender};
use crate::core::afora_error::AforaError;
use crate::features::pipeline::ports::subscriber_broadcast::{SubscriberBroadcast, SubscriberBuilder};
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::TrackingSubscriberInput;

pub struct ThreadedSubscriberBroadcaster {
    senders: Vec<Sender<Arc<TrackingSubscriberInput>>>,
    handles: Vec<JoinHandle<Result<(), AforaError>>>,
}

impl ThreadedSubscriberBroadcaster {
    pub fn new(builders: Vec<SubscriberBuilder>) -> Self {
        let mut senders = Vec::with_capacity(builders.len());
        let mut handles = Vec::with_capacity(builders.len());

        for builder in builders {
            let (tx, rx) = unbounded();
            senders.push(tx);

            handles.push(std::thread::spawn(move || -> Result<(), AforaError> {
                // La construcción sigue ocurriendo DENTRO del hilo,
                // exactamente como hoy. Nada cambia en cuanto a !Send.
                let mut subscriber = builder()?;
                while let Ok(event) = rx.recv() {
                    subscriber.notify_event(event)?;
                }
                Ok(())
            }));
        }

        Self { senders, handles }
    }
}

impl SubscriberBroadcast for ThreadedSubscriberBroadcaster {
    fn notify(&mut self, input: Arc<TrackingSubscriberInput>) {
        for sender in &self.senders {
            let _ = sender.send(input.clone());
        }
    }

    fn shutdown(&mut self) -> Result<(), AforaError> {
        self.senders.clear();
        for handle in self.handles.drain(..) {
            match handle.join() {
                Ok(inner) => inner?,
                Err(_) => return Err(AforaError::PostprocessError(
                    "Subscriber thread panicked".into(),
                )),
            }
        }
        Ok(())
    }
}

impl Drop for ThreadedSubscriberBroadcaster {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}