# This is a script to create VMs for the k8s cluster. DRAFT, check before running

# Create bridge network for the cluster

NET=kashef-albaraa

# solves network issue
incus network create $NET
iptables -I DOCKER-USER -i $NET -j ACCEPT
iptables -I DOCKER-USER -o $NET -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
iptables-save
