# 🎬 Roteiro da Demo — Redis LangCache

## 🎯 Contexto
**Company:** RedisLabs  
**Business Unit A:** Engenharia-de-Software — **Gabriel**  
**Business Unit B:** Financeiro — **Diego**

A demonstração mostra o isolamento do LangCache:
- por pessoa (função individual)
- por BU (contexto diferente)
- e por empresa (memória compartilhada entre idiomas)

---

## 🧩 Sequência de Perguntas

### 1️⃣ Identidade
**Gabriel:**  
> Minha função é Engenheiro de Software.  

**Diego:**  
> Minha função é Analista Financeiro.  

🗒️ *Cada cargo é salvo com isolamento individual.*

---

### 2️⃣ Recuperando o cargo
**Gabriel:**  
> Qual é a minha função na empresa?  

**Diego:**  
> Qual é a minha função na empresa?  

🗒️ *Mostra o cache por pessoa — cada um obtém seu cargo corretamente.*

---

### 3️⃣ Contexto por Business Unit
**Gabriel:**  
> O que significa o termo deploy?  

**Diego:**  
> O que significa o termo deploy?  

🗒️ *Mesma pergunta, sentidos diferentes — engenharia fala de código, financeiro fala de processo interno.*
🗒️ *Alterar no Redis a resposta do Diego*

---

### 4️⃣ Cache multilíngue (nível de empresa)
**Gabriel:**  
> Explique o que é aprendizado de máquina.  

**Diego:**  
> O que é machine learning?  

🗒️ *Perguntas em idiomas diferentes, mas equivalentes — cache hit entre PT-BR e EN.*

---

### 5️⃣ Cache global da empresa
**Ambos:**  
> O que é o Redis como um VectorDB, e por que ele ajuda a economizar tokens?  

🗒️ *Cache compartilhado ao nível da empresa. Mostra o poder do reuso global de respostas.*

---

🎯 **Dica para apresentação**
- Mostre o `[Cache Miss]` na primeira chamada.  
- Repita a pergunta e destaque o `[Cache Hit]`.  
- Feche com o insight: *“O Redis é mais que um banco, é um mecanismo inteligente pra proteger seus tokens.”*