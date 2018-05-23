# Toolbox de Identificação de Sistemas em Python

## Funções

### Mínimos Quadrados

*module_name.ls()*

**Parâmetros**

Parâmetros de entrada:

- *na* - Ordem do polinômio A(q^-1)= 1 + a1 * q^-1 + a2 * q^-2 + ... + a_na * q^-na

- *nb* - Ordem do polinômio B(q^-1)= b0 + b1 * q^-1 + b2 * q^-2 + ... + b_nb * q^-nb

- *u* - Dados de entrada

- *y* - Dados de saída

Parâmetros de saída:

- *theta* - Estimativa dos parâmetros do sistema

**Descrição do algoritmo**

Este algoritmo calcula a estimativa por mínimos quadrados para uma equação do
tipo A(q)y(t) = B(q)u(t) + e(t).

**Forma de Uso**

```
import <module_name>
(...)
```

[Exemplo Mínimos Quadrados](../examples/example_least_square.py)

--------------------------------------------------------------------------------

### Variáveis Instrumentais

*module_name.vi*

**Parâmetros**

Parâmetros de entrada:

- *na* - Ordem do polinômio A(q^-1)= 1 + a1 * q^-1 + a2 * q^-2 + ... + a_na * q^-na

- *nb* - Ordem do polinômio B(q^-1)= b0 + b1 * q^-1 + b2 * q^-2 + ... + b_nb * q^-nb

- *u* - Dados de entrada

- *y* - Dados de saída

- *u2* - Dados de entrada (Experimento adicional)

- *y2* - Dados de saída (Experimento adicional)

Parâmetros de saída:

- *theta()* - Estimativa dos parâmetros do sistema

**Descrição do algoritmo**

...

**Forma de Uso**

```
import <module_name>
(...)
```

[Exemplo Variáveis Instrumentais](../examples/example_instrumental_variable.py)

--------------------------------------------------------------------------------

### Mínimos Quadrados Recursivos

*module_name.rls*

**Parâmetros**

Parâmetros de entrada:

Parâmetros de saída:

**Descrição do algoritmo**

...

**Forma de Uso**

```
import <module_name>
(...)
```

[Exemplo Mínimos Quadrados Recursivos](../examples/example_recursive_least_square.py)

--------------------------------------------------------------------------------

### Steiglitz - Mc Bride

*module_name.smb*

**Parâmetros**

Parâmetros de entrada:

Parâmetros de saída:

**Descrição do algoritmo**

...

**Forma de Uso**

```
import <module_name>
(...)
```

[Exemplo Steiglitz - Mc Bride](../examples/example_steiglitz_mcbride.py)

--------------------------------------------------------------------------------