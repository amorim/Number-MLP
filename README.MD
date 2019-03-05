# Rede Neural MLP para identificar números

## Dados
A rede foi treinada para reconhecer os números 6, 5 e 2, correspondentes às iniciais do meu nome (L, O e A), conforme tabela abaixo:

![Tabela para conversão de números para letras](https://this-loli-have-a.pretty-pant.su/VzDYWNblyk.jpeg)

Os pesos e vieses da rede pré-treinada podem ser conferidos nessa [planilha](https://docs.google.com/spreadsheets/d/19OTK7q21ptjK1_ExsOBQQbCfRZqPML6jkRyViQbD1Cs/edit?usp=sharing).

Para facilitar, o script `driver.py` já vem configurado para carregar os dados da rede pré-treinada e gerar as imagens dos 3 números. Ao final da execução é exibida a acurácia do teste.

## Termo de garantia

A garantia dessa rede neural com os dados fornecidos é o acerto médio amortizado de 99,6% das imagens a ela fornecidas, seguindo as seguintes restrições:

- Fonte Arial ou Arial Bold tamanho 18;
- Resolução da imagem 28x28;
- Fundo branco e texto preto (podendo haver cinza proveniente do efeito de antialiasing);
- Números 6, 5 e 2;
- Rotação do número entre -45º e 45º, inclusivo;
- Translação do número a partir do centro em até 7 pixels.

Imagens que deixam de cumprir pelo menos uma dessas especificações não são qualificadas para a garantia de acerto médio amortizado de 99,6%.

## Gerar dados e treinar uma nova rede

Para facilitar a geração de imagens seguindo as restrições acima descritas, fornecemos um script para automatizar tal tarefa.

No script `driver.py` podem ser encontrados comentários que explicam como utilizar o script `dataset_generator.py`.

A rede não é limitada ao reconhecimento de 3 números, alterando o número de neurônios da camada de saída e configurando o gerador de forma adequanda, é possível utilizar qualquer conjunto de números entre 0 e 9. Lembrando que, nestes casos, o termo de garantia acima enunciado não é válido.
