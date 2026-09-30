# Idempotencia, reintentos y prevención de cobros duplicados

## Clave de idempotencia

Cada solicitud de pago debe incluir una clave de idempotencia estable por intención de negocio. Si el cliente repite la misma solicitud por timeout, el servicio recupera el resultado original en vez de crear un segundo cargo.

## Alcance

La clave debe asociarse al comercio, operación y payload relevante. Reutilizar la misma clave con datos incompatibles debe responder con conflicto. La ventana de retención depende de la política del producto y del ciclo de vida del pago.

## Reintentos

Los reintentos técnicos deben usar backoff y un máximo acotado. El código 91 puede ser reintentable; el código 05 no debe convertirse automáticamente en un bucle de reintentos. Los timeouts deben consultar el estado antes de volver a autorizar cuando el protocolo lo permita.

## Trazabilidad

El sistema debe registrar request id, idempotency key, payment id PAY-1008, estado previo, estado final y respuesta del proveedor para reconstruir incidentes.
