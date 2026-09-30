# Guía de autorización y códigos ISO 8583

## Flujo de autorización

Una transacción de compra entra por el Payment Gateway, pasa al adquirente y luego viaja por la red de pagos hasta el emisor. El emisor evalúa fondos, estado de la tarjeta, controles de riesgo y autenticación antes de responder.

## Código 05

El response code 05 significa "Do not honor". No identifica una causa única. Operaciones debe revisar la respuesta del emisor, reglas antifraude, estado de la tarjeta y trazas de autorización. No se debe reintentar en bucle porque puede aumentar rechazos y costos.

## Código 51

El response code 51 se asocia a fondos insuficientes. El comercio puede solicitar otro medio de pago o permitir un reintento posterior de acuerdo con su política.

## Código 91

El response code 91 suele indicar que el emisor o switch está temporalmente no disponible. Se puede aplicar un reintento controlado con backoff, siempre respetando idempotencia para no duplicar cargos.

## Timeout

Un timeout no equivale a un rechazo. Si el Payment Gateway no recibe la respuesta final del emisor, el estado debe quedar pendiente hasta consultar el resultado o ejecutar reverso según el protocolo.
