//! A typed service registry — the idiomatic Rust replacement for two Python
//! anti-patterns at once:
//!
//! * `service_registry.py`'s `ServiceRegistry` (a runtime type→impl map), and
//! * the pervasive `setattr(sys.modules["__main__"], ...)` monkey-patching in
//!   the `apply_*()` functions.
//!
//! Instead of mutating global module state at import time, components are
//! registered as `Arc<dyn Trait>` and injected explicitly. Lookups are by type,
//! thread-safe, and never panic on a missing entry (they return a typed error).

use std::any::{type_name, Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, SovereignError};

/// A thread-safe, type-indexed container of shared services.
///
/// ```
/// use sovereign_core::registry::ServiceRegistry;
/// trait Clock: Send + Sync { fn now(&self) -> u64; }
/// struct Fixed(u64);
/// impl Clock for Fixed { fn now(&self) -> u64 { self.0 } }
///
/// let mut reg = ServiceRegistry::new();
/// reg.register::<dyn Clock>(std::sync::Arc::new(Fixed(42)));
/// assert_eq!(reg.get::<dyn Clock>().unwrap().now(), 42);
/// ```
#[derive(Default)]
pub struct ServiceRegistry {
    // We store `Arc<T>` (or `Arc<dyn Trait>`) erased as `Arc<dyn Any>`.
    services: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for ServiceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The erased values aren't `Debug`; report the count of registrations.
        f.debug_struct("ServiceRegistry")
            .field("registered", &self.services.len())
            .finish()
    }
}

impl ServiceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
        }
    }

    /// Register an implementation for interface `T` (typically `dyn SomeTrait`).
    /// A later registration of the same `T` overwrites the earlier one.
    pub fn register<T: ?Sized + 'static>(&mut self, impl_: Arc<T>)
    where
        Arc<T>: Send + Sync,
    {
        self.services.insert(TypeId::of::<T>(), Box::new(impl_));
    }

    /// Fetch the implementation for interface `T`, or a typed error if absent.
    pub fn get<T: ?Sized + 'static>(&self) -> Result<Arc<T>>
    where
        Arc<T>: Clone + Send + Sync,
    {
        self.services
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<Arc<T>>())
            .cloned()
            .ok_or(SovereignError::ServiceMissing(type_name::<T>()))
    }

    /// Like [`get`](Self::get) but returns `None` instead of an error.
    pub fn get_optional<T: ?Sized + 'static>(&self) -> Option<Arc<T>>
    where
        Arc<T>: Clone + Send + Sync,
    {
        self.get::<T>().ok()
    }

    /// Whether interface `T` has a registered implementation.
    pub fn is_registered<T: ?Sized + 'static>(&self) -> bool {
        self.services.contains_key(&TypeId::of::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    trait Greeter: Send + Sync {
        fn greet(&self) -> String;
    }
    struct Hello;
    impl Greeter for Hello {
        fn greet(&self) -> String {
            "hi".into()
        }
    }

    #[test]
    fn register_and_resolve_trait_object() {
        let mut reg = ServiceRegistry::new();
        assert!(!reg.is_registered::<dyn Greeter>());
        reg.register::<dyn Greeter>(Arc::new(Hello));
        assert!(reg.is_registered::<dyn Greeter>());
        assert_eq!(reg.get::<dyn Greeter>().unwrap().greet(), "hi");
    }

    #[test]
    fn missing_service_is_typed_error() {
        let reg = ServiceRegistry::new();
        assert!(reg.get::<dyn Greeter>().is_err());
        assert!(reg.get_optional::<dyn Greeter>().is_none());
    }
}
