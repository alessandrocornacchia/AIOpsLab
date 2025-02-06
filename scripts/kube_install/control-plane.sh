#All below steps only for setting up controller

sudo systemctl enable --now kubelet
sudo kubeadm config images pull  --cri-socket /var/run/cri-dockerd.sock
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --cri-socket /var/run/cri-dockerd.sock
# sudo kubeadm init --pod-network-cidr=192.168.10.0/16 --control-plane-endpoint {`hostname --fqdn`,,} --cri-socket /var/run/cri-dockerd.sock
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config


# install network 
wget https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
kubectl apply -f kube-flannel.yml
sudo systemctl status kubelet --no-pager

sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --cri-socket /var/run/cri-dockerd.sock

# untaint the control plane to make it host pods 
#kubectl taint nodes <control-plane-node-name> node-role.kubernetes.io/control-plane:NoSchedule-