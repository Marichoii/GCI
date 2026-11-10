# v21 — cópia operacional para Colab

Preparação de 10/11/2026, sem nova versão de modelagem.

- [Notebook](gci_home_credit_v21_colab.ipynb)
- [Passo a passo](COMO_RODAR_NO_COLAB.md)
- [Diff legível das células](v21_colab.diff)
- [Auditoria de preservação](evidence/model_preservation.json)
- [Testes](evidence/test_report.json)
- [Wheels oficiais auditados](evidence/wheel_metadata.json)

Somente o cabeçalho e a célula original de ambiente mudaram. As outras
26 células originais foram preservadas integralmente, inclusive a célula
completa de treino e FROZEN_V21. Há adições de ambiente/entrada, cronometragem,
verificação e exportação. O diff compara o texto extraído das células, com
marcadores legíveis; não é um patch do JSON do notebook.

O notebook original validado, seu ZIP e seu CSV não foram substituídos.
Notebook original SHA-256: `3c3866074fe4ee9e6f83c036a390dd149e781e85b30eeb188e7580018103d367`.
Cópia Colab SHA-256: `382311bf0420a2bc1daf900eeb2c2c8b18d39668ce691a18ae5710856d250297`.
FROZEN_V21 SHA-256 canônico: `20610ca6bffd66b429dc5ab9a27ddcf320901a5fcf0a90c6b780a0b4b1e70608`.

Testes locais: 21 passaram. Foram verificadas as guardas de
reinício/instalação, hashes, upload/Drive com dublês, tempos/eventos de célula
sem executar o treino, CSV histórico e casos negativos. Foram resolvidas as
dependências Linux/Python 3.13 e 3.12 só com wheels, sem instalar pacotes.

Não executados: instalação real no Colab, transferências autenticadas
upload/Drive, treino novo, reprodução Linux/Python 3.13, benchmark de 90 min e
desconexão real. Nenhum CSV de submissão novo foi gerado. Os logs de teste
não são evidência de treinamento. A referência de CSV macOS continua com
SHA-256 `48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756`.

O arquivo requirements.txt é idêntico ao da v21. Os arquivos
resolved_linux_cp*.txt são auditorias de resolução e não substituem os pins.
O ZIP de entrega não contém dados, previsões, OOFs, modelos ou checkpoints.
