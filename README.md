# Transfer learning em domínio distante

Quanto de uma rede pré-treinada em ImageNet precisa ser descongelado para
classificar imagens de satélite?

Projeto final da disciplina de Machine Learning — Insper, 2026/2.

---

## A pergunta

Redes pré-treinadas em ImageNet são o ponto de partida padrão em visão
computacional. Só que o ImageNet é feito de fotografias de objetos, animais e
cenas do cotidiano, e imagem de satélite não se parece com nada disso: é vista
de cima, não tem objeto centralizado, e o que distingue uma classe da outra é
textura e padrão espacial em vez de forma.

As primeiras camadas de uma CNN aprendem coisas genéricas — borda, canto,
textura — que valem para qualquer imagem. As últimas aprendem combinações
específicas do domínio de origem, do tipo "orelha de cachorro" ou "roda de
carro", que não existem numa foto aérea de plantação.

Daí a pergunta: quanto da rede vale a pena descongelar, e existe um ponto em
que treinar do zero passa a ser melhor que reaproveitar o pré-treino?

## Os dados

[EuroSAT](https://github.com/phelber/EuroSAT): 27.000 imagens de satélite
Sentinel-2, RGB, 64×64, divididas em 10 classes de uso do solo (floresta,
rio, área industrial, plantação, pasto, entre outras).

Baixa automaticamente pelo `torchvision`, sem necessidade de conta ou download
manual:

```python
from torchvision.datasets import EuroSAT
EuroSAT(root="./data", download=True)
```

O treino é limitado a 2.000 imagens de propósito. Transfer learning existe
justamente para o regime de pouco dado — com as 27 mil imagens completas a
pergunta perde o sentido, porque qualquer arquitetura treina bem.

## O método

Três regimes, mesma divisão de treino e teste, mesma semente, mesmo número de
épocas:

| Regime | O que treina | Intuição |
| --- | --- | --- |
| Do zero | CNN pequena, todos os pesos | Ignora o pré-treino e aprende tudo com 2.000 imagens |
| Congelado | Só a cabeça de classificação | Usa as features do ImageNet como estão |
| Fine-tuning parcial | Cabeça + último bloco residual (`layer4`) | Mantém as features genéricas e readapta as específicas |

Backbone: `ResNet18` com pesos `IMAGENET1K_V1`. Importante: o pré-treino é em
ImageNet, nunca em EuroSAT. Existem modelos já fine-tunados em EuroSAT
disponíveis publicamente, e usar um deles invalidaria a pergunta, porque o
modelo já teria visto a tarefa que queremos medir.

No fine-tuning a taxa de aprendizado é reduzida (1e-4 contra 1e-3), para que os
pesos pré-treinados sejam ajustados e não destruídos.

**Regularização:** data augmentation com `RandomCrop` e `RandomHorizontalFlip`,
aplicado somente no conjunto de treino. A avaliação usa sempre o transform
determinístico — augmentar o teste mediria um problema diferente do que estamos
reportando.

**Métricas:** acurácia no teste, matriz de confusão, e o gap entre acurácia de
treino e de teste, que é o que mostra memorização.

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/experimento.ipynb
```

Ou direto no Colab pelo link no topo do notebook. O download do EuroSAT leva
alguns minutos na primeira execução. O experimento completo roda em cerca de
[TEMPO] numa GPU do Colab.

## Resultados

<!-- preencher depois de rodar -->

| Regime | Acurácia treino | Acurácia teste | Gap |
| --- | --- | --- | --- |
| Do zero | | | |
| Congelado | | | |
| Fine-tuning parcial | | | |

[Escrever aqui o que os números responderam à pergunta, incluindo o que
contrariou a expectativa inicial.]

## Estrutura

```
.
├── notebooks/
│   └── experimento.ipynb      # experimento completo, de ponta a ponta
├── relatorio.pdf              # relatório do projeto
├── model_card.md              # model card: uso pretendido, limites, vieses
├── requirements.txt
└── README.md
```

## Requisitos

```
torch
torchvision
matplotlib
numpy
scikit-learn
```

## Limitações

O EuroSAT é um dataset limpo, balanceado e de uma única fonte (Sentinel-2,
Europa). As conclusões sobre congelamento valem para esse recorte e não devem
ser generalizadas para outros sensores, outras regiões ou imagens de resolução
diferente. Mais detalhes no `model_card.md`.

## Vídeo

[link para o vídeo de 5 minutos]

## Autores

[Nome 1] e [Nome 2]
