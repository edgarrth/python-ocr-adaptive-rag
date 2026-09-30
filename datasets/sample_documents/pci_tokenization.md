# PCI DSS, PAN y tokenización

## PAN

El Primary Account Number es dato de cuenta sensible para el entorno de pagos. Su almacenamiento y transmisión deben reducirse al mínimo y protegerse de acuerdo con los controles aplicables de PCI DSS.

## Tokenización

La tokenización reemplaza el PAN por un token que no expone directamente el número de tarjeta. Permite reducir la presencia del PAN en servicios de negocio y limita el alcance de componentes que necesitan manejar el dato real.

## CVV

El código de verificación no debe conservarse como dato de autenticación sensible después de la autorización. La PoC usa esta regla como ejemplo de consulta documental y no como sustituto de una evaluación formal de cumplimiento.

## Arquitectura

El frontend y los servicios de negocio trabajan con tokens. Solo el componente de vault o proveedor de tokenización mantiene la relación con el PAN. El Payment Gateway consume el token y evita propagar el PAN a componentes que no lo necesitan.
