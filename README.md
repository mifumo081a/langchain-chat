# docker-lab

## What is this

次を内包する実験・デモ環境を提供するDocker環境です。
- jupyter lab
- streamlit
- pytorch with CUDA

# Contents
- [How to use](#sec1)
    - [Preparations](#subsec1-1)
    - [Start project](#subsec1-2)
    - [別マシンのブラウザからjupyterへアクセスする方法](#subsec1-3)
- [Appendix](#sec2)
    - [Trouble shooting](#subsec2-1)
    - [Docker document](#subsec2-2)
    - [GPU Machine setup Ubuntu 22.04](#subsec2-3)
- [References](#sec3)

<a id="sec1"></a>
# How to use
<a id="subsec1-1"></a>
## Preparations
1. Docker、docker-composeをインストールする[Link](./docs/docker_doc.md)
    - docker compose(ver1.28.0>=)をインストールする（Mac, Windowsの場合はDockerインストール時に入ってる）
2. **次のコマンドを実行して`.env`ファイルを作成する**
    - For Windows
        ```sh
        $ bash ./scripts/setDotEnv.sh
        ```
    - For Linux or Mac
        ```sh
        $ sh ./scripts/setDotEnv.sh
        ```
3. GPUを使用するなら次を実行する
    - GPUドライバをインストールする
        - [Ubuntuの場合](./docs/ubuntu2204_GPU_machine_setup.md)
    - **Nvidia Container Toolkit**をセットアップする
        - For Windows
            WSLでgpuSetup.shを実行する
            ```sh
            $ bash ./scripts/install-nvidia-container-toolkit.sh
            ```
        - For Linux or Mac
            gpu_setup.shを実行する
            ```sh
            $ sh ./scripts/install-nvidia-container-toolkit.sh
            ```
    - 完了したらDocker Containerでnvidia-smiを実行して確認する
        ```sh
        $ sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
        ```

<a id="subsec1-2"></a>
## Start project
1. 作業するファイル等を`/workspace`内に移動してください
2. build image
    - With GPU
        ```sh
        $ docker-compose -f gpu.yml build
        ```

    - Without GPU
        ```sh
        $ docker-compose build
        ```
3. containerを起動する
    - With GPU
        ```sh
        $ docker-compose -f gpu.yml up
        ```

    - Without GPU
        ```sh
        $ docker-compose up
        ```
4. サービス指定
- jupyter lab のみ
    上記のコマンドに次のように書き足し、ブラウザからlocalhost:8888にアクセス
    ```sh
    $ ~~up jupyterlab
    ```

- streamlit のみ
    上記のコマンドに次のように書き足し、ブラウザからlocalhost:8501にアクセス
    ```sh
    $ ~~up streamlit
    ```
5. 好きなライブラリをインストールする
    jupyterlabのセル内で次のようにすれば良い
    ```python
    !pip install [library]
    ```

<a id="subsec1-3"></a>
## 別マシンのブラウザからjupyterへアクセスする方法
1. ip:ポート番号で検索する
    - 8080等のポートを解法する
    - jupyter等でパスワードを設定する（任意）
    - コンテナを起動する
    - `ip address`等でリモートホストのipアドレスを調べる
    - `http://[ipアドレス]:8080`でアクセスする
2. ssh
    - リモートホストにsshサーバーを構築する
    - ローカルから次のコマンドを実行する
        ```sh
        $ ssh -L 8080:localhost:8080 <hostname>@<hostip>
        ```
    - ローカルのブラウザから`localhost:8080`にアクセスすると、jupyterを実行することができる。


<a id="sec2"></a>
# Appendix
<a id="subsec2-1"></a>
## [Trouble shooting](./docs/trouble_shooting.md)
問題が発生した場合に読んでください。

<a id="subsec2-2"></a>
## [Docker document](./docs/docker_doc.md)
docker関係のドキュメントです

<a id="subsec2-3"></a>
## [GPU Machine setup Ubuntu 22.04](./docs/ubuntu2204_GPU_machine_setup.md)

<a id="sec3"></a>
# References
1. [入門 Docker](https://y-ohgi.com/introduction-docker/)
2. [Docker documents ja, WindowsへのDocker Desktopのインストール](https://docs.docker.jp/docker-for-windows/install.html)
1. [Zenn, WSL2+Docker DesktopでPytorchのGPU環境を構築する](https://zenn.dev/takeguchi/articles/361e12a5321095)
2. [Zenn, docker-comopseで環境によってDockerfileのCMDを使い分けたい](https://zenn.dev/akira_kashihara/articles/073b4b19a13840)
3. [Qiita, docker-composeでGPU環境(+PyTorch)を構築する](https://qiita.com/Sicut_study/items/32eb5dbaec514de4fc45)
4. [Nvidia, nvidia-docker Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
5. [Nvidia, nvidia-docker Troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/troubleshooting.html)
6. [Blog, docker「docker.errors.DockerException: Error while fetching server API version:」が発生した場合の対処法](https://mebee.info/2021/10/13/post-44471/)
7. [Blog, jupyter notebookの「既に使用されて」いるポートを開放する](https://life-is-miracle-wind.blog.jp/archives/30965602.html)
8. [Qiita, Docker Composeのvolume使用時に出会うpermission on deniedに対応する一つの方法](https://qiita.com/cheekykorkind/items/ba912b62d1f59ea1b41e)
9. [Blog, リモートのDocker上のjupyterにポートフォワーディング](https://no-retire-no-life.com/ec2-docker-jupyter/)
