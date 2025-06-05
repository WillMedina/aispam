// Función para enviar datos por POST de manera asíncrona
async function enviarTexto(url, data) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        //console.log('Resultado:', result);
        return result;

    } catch (error) {
        //console.log('Error:', error);
        throw error;
    }
}

// Función que se ejecutará al presionar el boton
async function procesarTexto() {
    const textareaElement = document.getElementById('txtEmail');
    const textareaValue = textareaElement.value;
    const mensaje = document.getElementById('msjFrmTexto');
    const buttonElement = document.getElementById('btnEnviarTexto');
    const resultadoDiv = document.getElementById('Resultado');

    // Deshabilitar el botón y hacer el textarea de solo lectura
    buttonElement.disabled = true;
    textareaElement.readOnly = true;

    // Mostrar el mensaje "Cargando resultados..."
    mensaje.innerHTML = '<b><i class="fas fa-spinner fa-spin"></i> Cargando resultados...</b>';

    try {
        const result = await enviarTexto('apiTexto.php', {texto: textareaValue});
        //console.log('Resultado recibido:', result);

        // Ocultar el modal de texto y mostrar el modal de resultados
        $('#frmTexto').modal('hide');
        $('#frmMensaje').modal('show');

        // Mostrar los resultados en el modal de resultados
        let resultados_obtenidos = result;
        let texto_final = '';
        if (resultados_obtenidos.resultado == '1') {
            /* let porcentaje = (resultados_obtenidos.datos.confidence * 100).toFixed(2) + "%"; */
            let porcentaje = (resultados_obtenidos.datos.prediction_probability_spam * 100).toFixed(2) + "%";
            texto_final = '<b>Los resultados obtenidos son:</b><br>';
            /*if (resultados_obtenidos.datos.spam) { */
            if (resultados_obtenidos.datos.is_spam) {
                texto_final += '<b>Spam detectado: <span style="color:red">S&Iacute;</span></b><br>';
            } else {
                texto_final += '<b>Spam detectado: <span style="color:green">NO</span></b><br>';
            }

            texto_final += '<b>An&aacute;lisis de SPAM: <span style="color:blue">' + porcentaje + '</span></b><br>';
            texto_final += '<b>Umbral de medida: <span style="color:blue">0.5 (o 50%)</span></b><br>';
        } else {
            texto_final = '<b>Error ocurrido procesando los datos. Vuelva a intentarlo m&aacute;s tarde.</b>';
            console.log(result);
        }

        resultadoDiv.innerHTML = texto_final;

        // Limpiar el textarea
        textareaElement.value = '';
        mensaje.innerHTML = '';
        buttonElement.disabled = false;
        textareaElement.readOnly = false;
    } catch (error) {
        console.log('Error al procesar los datos:', error);

        // Ocultar el mensaje y habilitar el botón y el textarea
        mensaje.innerHTML = '';
        buttonElement.disabled = false;
        textareaElement.readOnly = false;
    }
}

// Agrega un evento de clic al botón para ejecutar la función
const buttonElement = document.getElementById('btnEnviarTexto');
buttonElement.addEventListener('click', procesarTexto);

// Función para enviar archivo EML
async function enviarArchivoEML() {
    let fileInput = document.getElementById('file');
    let formData = new FormData();
    let mensaje = document.getElementById('msjFrmEML');
    formData.append('file', fileInput.files[0]);

    mensaje.innerHTML = '<b><i class="fas fa-spinner fa-spin"></i> Subiendo archivo y cargando resultados...</b>';
    try {
        const response = await fetch('apiEML.php', {
            method: 'POST',
            body: formData
                    /*
                     onUploadProgress: (progressEvent) => {
                     console.log('Progreso:', progressEvent);
                     if (progressEvent.lengthComputable) {
                     const percentComplete = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                     mensaje.innerHTML = `<b><i class="fas fa-spinner fa-spin"></i> Subiendo archivo (${percentComplete}%) y cargando resultados...</b>`;
                     }
                     } */
        });

        const result = await response.json();

        let resultados_obtenidos = result;
        let texto_final = '';
        if (resultados_obtenidos.resultado == '1') {
            /* let porcentaje = (resultados_obtenidos.datos.confidence * 100).toFixed(2) + "%"; */
            let porcentaje = (resultados_obtenidos.datos.prediction_probability_spam * 100).toFixed(2) + "%";
            texto_final = '<b>Los resultados obtenidos son:</b><br>';
            /*if (resultados_obtenidos.datos.spam) { */
            if (resultados_obtenidos.datos.is_spam) {
                texto_final += '<b>Spam detectado: <span style="color:red">S&Iacute;</span></b><br>';
            } else {
                texto_final += '<b>Spam detectado: <span style="color:green">NO</span></b><br>';
            }

            texto_final += '<b>An&aacute;lisis de SPAM: <span style="color:blue">' + porcentaje + '</span></b><br>';
            texto_final += '<b>Umbral de medida: <span style="color:blue">0.5 (o 50%)</span></b><br>';
        } else {
            texto_final = '<b>Error ocurrido procesando los datos. Vuelva a intentarlo m&aacute;s tarde.</b>';
            console.log(result);
        }


        const resultadoDiv = document.getElementById('Resultado');
        resultadoDiv.innerHTML = texto_final;

        mensaje.innerHTML = '';

        $('#frmEML').modal('hide');
        $('#frmMensaje').modal('show');
    } catch (error) {
        console.error('Error al enviar el archivo EML:', error);
        mensaje.innerHTML = 'Error en la subida de archivo, pruebe a refrescar el sitio.';
    }
}


// Agrega un evento de clic al botón de subir archivo EML
const btnSubirEML = document.getElementById('btnSubirEML');
btnSubirEML.addEventListener('click', enviarArchivoEML);
