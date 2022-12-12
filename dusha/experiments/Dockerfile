# Distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:10.1-base-ubuntu18.04
RUN apt-get update && apt-get install -y git && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3.8 && apt install -y python3-pip && \
    pip3 install virtualenv && \
    virtualenv -p python3.8 --no-download venv

SHELL ["/bin/bash", "-c"]
RUN source /venv/bin/activate && \
    pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install scikit-learn==0.24.0 librosa==0.7.1 tqdm==4.55.1 click==8.1.2 pandas==1.1.5 tensorboard==2.4.1 crowd-kit==1.0.0 && \
    apt install -y libsndfile1 && pip install numba==0.48

WORKDIR /packages

RUN git clone https://github.com/maxme1/lazycon.git
WORKDIR lazycon
RUN git checkout tags/v0.3.1

RUN source /venv/bin/activate && pip install -e .
WORKDIR /workspace

CMD ["bash"]
