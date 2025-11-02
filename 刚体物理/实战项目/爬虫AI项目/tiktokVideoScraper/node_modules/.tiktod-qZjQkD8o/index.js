const axios = require("axios");
const Utils = require("./lib/utils");

//Download Video / Images 
function download(url) {
  return new Promise((resolve, reject) => {
    if (!url) return reject(new Error("url input is required"));
    axios.get(Utils.TOD + "/download", { params: { url } })
      .then((dl) => resolve(dl.data))
      .catch(reject);
  });
}

//Get User info
function stalk(username) {
  return new Promise((resolve, reject) => {
    if (!url) return reject(new Error("username is required"));
    axios.get(Utils.TOD + "/stalk", { params: { username } })
      .then((stalker) => resolve(stalker.data))
      .catch(reject);
  });
}

//Random Porn Tiktok?
function porn() {
  return new Promise((resolve, reject) => {
    axios.get(Utils.TOD + "/porn")
      .then((porner) => resolve(porner.data))
      .catch(reject);
  });
}

module.exports = { download, stalk, porn };
