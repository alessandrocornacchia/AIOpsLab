
# install Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm

###################################
#### Worker
# Replace with YOUR HASH for cert
sudo kubeadm join <ip>:6443 --token <token> --discovery-token-ca-cert-hash <cert_hash> --cri-socket unix:///var/run/cri-dockerd.sock
mkdir -p $HOME/.kube
# copy the control plane admin.conf to the worker's .kube/ccloonfig
# In control plane
sudo cat /etc/kubernetes/admin.conf 
# In worker
nano $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# test
kubectl get nodes