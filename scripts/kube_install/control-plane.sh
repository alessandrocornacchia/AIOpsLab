#All below steps only for setting up controller

CIDR_NET=172.41.0.0/16

sudo systemctl enable --now kubelet
sudo kubeadm config images pull  --cri-socket /var/run/cri-dockerd.sock
# sudo kubeadm init --pod-network-cidr=172.40.0.0/16 --cri-socket /var/run/cri-dockerd.sock
sudo kubeadm init --pod-network-cidr=$CIDR_NET --control-plane-endpoint `hostname --fqdn` --cri-socket /var/run/cri-dockerd.sock
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config


# install network 
# Needs manual creation of namespace to avoid helm error
kubectl create ns kube-flannel
kubectl label --overwrite ns kube-flannel pod-security.kubernetes.io/enforce=privileged

helm repo add flannel https://flannel-io.github.io/flannel/
helm install flannel --set podCidr="$CIDR_NET" --namespace kube-flannel flannel/flannel

# sudo kubeadm init --pod-network-cidr=172.40.0.0/16 --cri-socket /var/run/cri-dockerd.sock

# untaint the control plane to make it host pods 
kubectl taint nodes `hostname --fqdn` node-role.kubernetes.io/control-plane:NoSchedule-