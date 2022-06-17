const selectProducts = document.getElementById("select-products");
const selectScores = document.getElementById("select-score");
const qtyInput = document.getElementById("input-qty");
const delayInput = document.getElementById("input-delay");
const info = document.getElementById("info");
const picturesCollection = document.getElementById('pictures-collection');
const infoEmpty = "Select product, score and at least one picture";
info.innerHTML=infoEmpty;
var productText = null;
var scoreText = null;
var productAlias = null;
var scoreNumber = null;
var fileNames = [];
var selectedFileNames = [];

const onCapture = () => {
    checkRemove('Esto eliminara las imágenes ya capturadas');
    capture()
    console.warn("Imagenes capturadas");
}

const onRemove = () => {
    checkRemove();
}

const onUpload = (e) => {
    e.preventDefault();
    upload();
    clearImages();
}

const drawInfo = () => {
    if (selectProducts.value && selectScores.value && fileNames.length) {
        productText = selectProducts.item(selectProducts.selectedIndex).innerText;
        scoreText = selectScores.item(selectScores.selectedIndex).innerText;
        productAlias = selectProducts.value;
        scoreNumber = selectScores.value;
        info.innerHTML = `<b>Product:</b> ${productText}<br> <b>Score:</b> ${scoreText}`;
        console.log(productAlias + '/' + scoreNumber);
        document.getElementById("upload-button").disabled = false;
    }
    else {
        info.textContent = infoEmpty;
        document.getElementById("upload-button").disabled = true;
    }
};

const upload = () => {
    if (confirm(`ProductId=${productAlias}\nScoreId=${scoreNumber}\nSend?`)) {
        console.log(`Sent: ProductId=${productAlias} | ScoreId=${scoreNumber}`)
        let formData = {
            productAlias,
            scoreNumber,
            fileNames
        };
        console.log('Sent data: ', formData);
        let header = new Headers(
            {
                'X-CSRFToken': getCookie('csrftoken'),
                'Content-Type': 'application/json'
            })
        formData = JSON.stringify(formData);
        fetch('/upload/', {
            method: 'POST',
            headers: header,
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                console.log('Received data: ', data)
            })
            .catch(error => {
                console.error('Error:', error);
            });
    };
}

document.getElementById('upload-form').addEventListener('submit', onUpload);

const capture = () => {
    let header = new Headers(
        {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json'
        })

    fetch('/capture?' + new URLSearchParams(
        {
            qty: parseInt(qtyInput.value) || 1,
            delay: parseInt(delayInput.value) || 0
        }), {
        method: 'GET',
        headers: header
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            fileNames = data['pics'];
            placePictures(fileNames);
        })
        .catch(error => console.error('Error:', error));
}

const placePictures = (pics) => {
    pics.forEach((pic) => {
        const picture = document.createElement('img');
        picture.src = 'static' + pic;
        picturesCollection.appendChild(picture);
    });
    drawInfo();
}

const removePicsFromStorage = (pics) => {
    let header = new Headers(
        {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json'
        })

    let formData = { fileNames: pics };
    formData = JSON.stringify(formData);

    fetch('/remove/', {
        method: 'DELETE',
        headers: header,
        body: formData,
        keepalive: true
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
        })
        .catch(error => console.error('Error:', error));
}

const clearImages = () => {
    picturesCollection.innerHTML = '';
    fileNames = [];
    selectedFileNames = [];
    console.warn("Imagenes removidas");
}

const checkRemove = (message) => {
    if (fileNames.length && confirm(message || "Remover las imágenes capturadas?")) {
        removePicsFromStorage(fileNames);
        clearImages();
    }
}

const getCookie = (name) => {
    let cookies = {};
    document.cookie.split(';').forEach((cookie) => {
        let [key, val] = cookie.split('=');
        cookies[key.trim()] = val;
    });
    return cookies[name];
}

const clearPicsWhenLeaving = () => {
    if (fileNames.length){
        removePicsFromStorage(fileNames)
    }
}

window.onunload = clearPicsWhenLeaving