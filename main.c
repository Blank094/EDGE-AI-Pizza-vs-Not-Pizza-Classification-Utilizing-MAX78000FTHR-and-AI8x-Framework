/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All rights Reserved.
*
* Enhanced pizza-notpizza with Camera, Button, and ASCII Art support
* Board: FTHR_REVA (Feather Board)
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_sys.h"
#include "icc.h"
#include "led.h"
#include "dma.h"
#include "pb.h"
#include "cnn.h"
#include "mxc_delay.h"
#include "camera.h"

// Comment out to disable ASCII art display
#define ASCII_ART

// Image dimensions (64x64 for pizza model)
#define IMAGE_SIZE_X (64)
#define IMAGE_SIZE_Y (64)

#define CAMERA_FREQ (5 * 1000 * 1000)

const char classes[CNN_NUM_OUTPUTS][12] = { "Pizza", "Not Pizza" };

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time; // Stopwatch

// Buffer for camera image
static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y];

/* **************************************************************************** */
#ifdef ASCII_ART
// ASCII brightness characters from dark to light
char *brightness = "@%#*+=-:. "; // simple
#define RATIO 1 // ratio of scaling down the image (1 = no scaling for 64x64)

void asciiart(uint8_t *img)
{
    int skip_x, skip_y;
    uint8_t r, g, b, Y;
    uint8_t *srcPtr = img;
    int l = strlen(brightness) - 1;

    skip_x = RATIO;
    skip_y = RATIO;
    
    printf("\n=== ASCII Art Representation ===\n");
    
    for (int i = 0; i < IMAGE_SIZE_Y; i++) {
        for (int j = 0; j < IMAGE_SIZE_X; j++) {
            // 0x00bbggrr, convert to [0,255] range
            r = *srcPtr++ ^ 0x80;
            g = *(srcPtr++) ^ 0x80;
            b = *(srcPtr++) ^ 0x80;
            srcPtr++; //skip msb=0x00

            // Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            Y = (3 * r + b + 4 * g) >> 3; // simple luminance conversion
            
            if ((skip_x == RATIO) && (skip_y == RATIO))
                printf("%c", brightness[l - (Y * l / 255)]);

            skip_x++;
            if (skip_x > RATIO)
                skip_x = 1;
        }
        skip_y++;
        if (skip_y > RATIO) {
            printf("\n");
            skip_y = 1;
        }
    }
    printf("================================\n\n");
}
#endif

/* **************************************************************************** */
void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1) {}
}

/* **************************************************************************** */
void cnn_load_input(void)
{
    int i;
    const uint32_t *in0 = input_0;

    for (i = 0; i < 4096; i++) {
        // Wait for FIFO 0
        while (((*((volatile uint32_t *)0x50000004) & 1)) != 0) {}
        *((volatile uint32_t *)0x50000008) = *in0++; // Write FIFO 0
    }
}

/* **************************************************************************** */
void capture_process_camera(void)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;
    int cnt = 0;
    uint8_t r, g, b;
    uint8_t *data = NULL;
    stream_stat_t *stat;

    printf("Starting camera capture...\n");
    camera_start_capture_image();

    // Get the details of the image from the camera driver.
    camera_get_image(&raw, &imgLen, &w, &h);
    printf("Camera: W=%d H=%d Length=%d\n", w, h, imgLen);

    // Get image line by line
    for (int row = 0; row < h; row++) {
        // Wait until camera streaming buffer is full
        while ((data = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
        }

        for (int k = 0; k < 4 * w; k += 4) {
            // data format: 0x00bbggrr
            r = data[k];
            g = data[k + 1];
            b = data[k + 2];
            //skip k+3

            // change the range from [0,255] to [-128,127] and store in buffer for CNN
            input_0[cnt++] = ((b << 16) | (g << 8) | r) ^ 0x00808080;
        }

        // Release stream buffer
        release_camera_stream_buffer();
    }

    stat = get_camera_stream_statistic();

    if (stat->overflow_count > 0) {
        printf("ERROR: Camera overflow detected = %d\n", stat->overflow_count);
        LED_On(LED2); // Turn on red LED if overflow detected
        while (1) {}
    }
    
    printf("Camera capture complete!\n");
}

/* **************************************************************************** */
int main(void)
{
    int i;
    int digs, tens;
    int ret = 0;
    int result[CNN_NUM_OUTPUTS];
    int dma_channel;

    // Wait for PMIC 1.8V to become available
    MXC_Delay(200000);
    
    printf("\n\n=================================\n");
    printf("Pizza-vs-NotPizza Feather Demo\n");
    printf("=================================\n\n");

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Switch to 100 MHz clock */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Enable peripheral, enable CNN interrupt, turn on CNN clock */
    /* CNN clock: 50 MHz div 1 */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    /* Configure P2.5, turn on the CNN Boost */
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);

    /* Bring CNN state machine into consistent state */
    cnn_init();
    /* Load CNN kernels */
    cnn_load_weights();
    /* Load CNN bias */
    cnn_load_bias();
    /* Configure CNN state machine */
    cnn_configure();

    // Initialize DMA for camera interface
    printf("Initializing DMA...\n");
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    // Initialize camera
    printf("Initializing Camera...\n");
    camera_init(CAMERA_FREQ);

    ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, 
                       STREAMING_DMA, dma_channel);
    if (ret != STATUS_OK) {
        printf("Error: Camera setup failed with error %d\n", ret);
        return -1;
    }

    // Set camera clock prescaler to prevent streaming overflow
    camera_write_reg(0x11, 0x0);
    printf("Camera initialized successfully!\n\n");

    printf("********** Press PB1(SW1) to capture an image **********\n");
    while (!PB_Get(0)) {}

    // Enable CNN clock
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

    printf("\n*** CNN Inference Started ***\n\n");

    while (1) {
        LED_Off(LED1);
        LED_Off(LED2);

        // Capture image from camera
        capture_process_camera();

        printf("Starting CNN inference...\n");
        cnn_start();
        cnn_load_input();

        SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
        while (cnn_time == 0) {
            __WFI(); // Wait for CNN interrupt
        }

        // Unload CNN data
        cnn_unload((uint32_t *)ml_data);
        cnn_stop();

        // Softmax
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);

        printf("\nInference Time: %d us\n", cnn_time);
        printf("\n*** Classification Results ***\n");

        for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
            digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
            tens = digs % 10;
            digs = digs / 10;
            result[i] = digs;
            printf("[%7d] -> Class %d (%10s): %d.%d%%\n", 
                   ml_data[i], i, classes[i], result[i], tens);
        }

        // Determine winner and control LEDs
        if (result[0] == result[1]) {
            printf("\nResult: Unknown (Tie)\n");
            LED_On(LED1);
            LED_On(LED2);
        } else if (ml_data[0] > ml_data[1]) {
            printf("\nResult: %s detected with %d.%d%% confidence\n", 
                   classes[0], result[0], tens);
            LED_On(LED1);  // Pizza detected
            LED_Off(LED2);
        } else {
            printf("\nResult: %s detected with %d.%d%% confidence\n", 
                   classes[1], result[1], tens);
            LED_Off(LED1);
            LED_On(LED2);  // Not Pizza detected
        }

#ifdef ASCII_ART
        asciiart((uint8_t *)input_0);
#endif

        printf("\n********** Press PB1(SW1) to capture next image **********\n");
        while (!PB_Get(0)) {}
        printf("\n");
    }

    return 0;
}