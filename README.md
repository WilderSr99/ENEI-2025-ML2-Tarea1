
# Assignment 1 - Bag of Words, SVM y Árboles de Regresión  

**Curso:** *2025-G1-910040-4-PEUCD-MACHINE LEARNING II*  
---
## Integrantes del grupo
	- Buleje Ticse, Jean Carlos
	- Rosales Chuco, Noel Ivan
	- Sebastian Rios, Wilder Teddy

---

## Descripción general  

Este trabajo aplica **tres técnicas supervisadas de aprendizaje automático**:  
- **Parte A:** Clasificación binaria de texto (*Disaster Tweets*).  
- **Parte B:** Análisis del **sobreajuste en SVM**.  
- **Parte C:** **Árboles de regresión** para predecir ventas (*Carseats*).  

Los **procedimientos detallados, código y visualizaciones** se encuentran en la carpeta **scr** con los archivos `.ipynb` del repositorio.

---

## Parte A — Clasificación de Tweets  
**Objetivo:** predecir si un tweet corresponde a un desastre real.  
- Representación: *Bag of Words binario* (292 palabras).  
- Modelos: Regresión Logística (sin, L1, L2) y Bernoulli NB.  
- **Mejor modelo:** *Logistic Regression L1* → *F1 ≈ 0.70*.  
- Palabras más asociadas a desastres: `spill`, `debris`, `bomber`.  

*scr:* `ParteA_Logistic_Binaria.ipynb`

---

## Parte B — SVM y parámetro C  
**Objetivo:** estudiar el efecto del parámetro `C` en el desempeño y sobreajuste del modelo.  
- Datos simulados con dos clases apenas separables.  
- Errores mínimos (~4%) para valores de **C entre 1 y 10**.  
- **C bajo:** modelo simple → *underfitting*.  
- **C alto:** modelo complejo → *sobreajuste*.  

 *scr:* `ParteB_SVM_Overfitting.ipynb`

---

## Parte C — Árboles de Regresión (*Carseats*)  
**Objetivo:** predecir `Sales` (ventas) a partir de variables como `Price`, `ShelveLoc`, `Advertising`, `Age` e `Income`.  
- Árbol base: *MSE train = 0.000*, *MSE test = 4.35* → sobreajuste total.  
- Árbol podado (`ccp_alpha = 0.0493`): *MSE train = 1.45*, *MSE test = 4.27* → mejor equilibrio.  
- Variables más influyentes: **`ShelveLoc`** (ubicación del producto) y **`Price`** (precio).  
 *scr:* `ParteC_Arbol_Regresion.ipynb`

---

## Conclusión general  
- La **regularización** (L1/L2), la **poda** y el **ajuste de hiperparámetros** ayudan a controlar el sobreajuste y mejorar la generalización.  
- Los tres ejercicios evidencian el principio central del aprendizaje supervisado:  
  > *equilibrar el sesgo y la varianza para obtener modelos robustos y explicativos.*  

 *Los resultados completos y gráficos se encuentran en los notebooks `.ipynb` del repositorio.*

---
 *Elaborado como parte del Assignment 1 — ENEI-2025-ML2: Machine Learning II.*
