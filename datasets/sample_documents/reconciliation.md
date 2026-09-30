# Conciliación y liquidación de pagos

## Relación con la autorización

La autorización confirma que el emisor acepta reservar o comprometer fondos, pero no completa la liquidación. Luego el adquirente genera archivos o eventos de clearing que se contrastan con las transacciones registradas por el Payment Gateway.

## Transacción autorizada ausente en conciliación

Cuando una transacción fue autorizada y no aparece en conciliación, operaciones debe validar primero que exista captura, revisar el identificador PAY de la operación, confirmar que fue incluida en el lote de clearing y verificar rechazos del adquirente. También debe revisar reversos, anulaciones y duplicados.

## Componentes

Payment Gateway -> Adquirente -> Red de pagos -> Emisor participa en autorización. Para conciliación intervienen Payment Gateway, ledger operativo, adquirente, archivos de clearing y proceso de settlement.

## Controles

La correlación debe usar identificadores estables y no solo monto y fecha. Los reintentos del envío de clearing deben ser idempotentes. Un registro de auditoría debe conservar la transición entre autorizado, capturado, conciliado y liquidado.
