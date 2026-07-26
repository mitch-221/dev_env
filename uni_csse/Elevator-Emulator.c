/*
 * Elevator-Emulator.c
 *
 * Main file with Safe SSD Display Implementation and Joystick Service Mode
 *
 * Authors: Peter Sutton, Ahmed Baig
 * Modified by: Mitchell Wishart
 */ 

/* Definitions */

#define F_CPU 8000000L

/* External Library Includes */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <stdio.h>
#include <util/delay.h>
#include <stdbool.h>

/* Internal Library Includes */

#include "display.h"
#include "ledmatrix.h"
#include "buttons.h"
#include "serialio.h"
#include "terminalio.h"
#include "timer0.h"
#include "pixel_colour.h"

#define SSD_SEG_A   PA3
#define SSD_SEG_B   PA2   
#define SSD_SEG_C   PA1  
#define SSD_SEG_D   PA0 
#define SSD_SEG_E   PD2  
#define SSD_SEG_F   PD7  
#define SSD_SEG_G   PD6  
#define CC_PIN      PD4
#define SSD_DP      PD5
#define LED_L0      PC4
#define LED_L1      PC5  
#define LED_L2      PC6
#define LED_L3      PC7

#define MAX_TRAVELLERS 10
#define abs(x) ((x) < 0 ? -(x) : (x))

// JOYSTICK SERVICE MODE DEFINITIONS
#define JOYSTICK_Y_PIN      5
#define SERVICE_MODE_SW     PA4
#define JOYSTICK_CENTER     512
#define JOYSTICK_DEADZONE   50
#define JOYSTICK_SLOW_THRESHOLD  150
#define JOYSTICK_FAST_THRESHOLD  300

/* Data Structures */

typedef enum {UNDEF_FLOOR = -1, FLOOR_0=0, FLOOR_1=4, FLOOR_2=8, FLOOR_3=12} ElevatorFloor;

typedef struct {
    ElevatorFloor start_floor;    
    ElevatorFloor dest_floor;     
    bool active;                  
    uint8_t queue_position;     
} QueuedTraveller;

/* Global Variables */
uint32_t time_since_move;
ElevatorFloor current_position;
ElevatorFloor destination;
static int8_t stored_direction = 0;
static ElevatorFloor stored_floor = FLOOR_0;
uint16_t floors_with_traveller = 0;
uint16_t floors_without_traveller = 0;

/* Sound and Door Variables */
uint32_t sound_start_time = 0;
uint8_t sound_playing = 0;
uint16_t sound_duration = 0;
volatile uint8_t sound_toggle = 0;
uint8_t door_animation_active = 0;
uint32_t door_animation_start_time = 0;
uint8_t door_animation_phase = 0;   

QueuedTraveller travellers[MAX_TRAVELLERS]; // Array of all travellers
int8_t current_traveller_index = -1;
uint8_t total_travellers = 0;
int8_t target_traveller_index = -1;

// JOYSTICK SERVICE MODE VARIABLES
uint8_t service_mode_active = 0;  // 0 = automatic, 1 = service mode
uint16_t joystick_value = 512;

/* Internal Function Declarations */

void initialise_hardware(void);
void start_screen(void);
void start_elevator_emulator(void);
void handle_inputs(void);
void draw_elevator(void);
void draw_floors(void);
ElevatorFloor read_destination_from_switches(void);
void check_floor_movement(ElevatorFloor old_position, ElevatorFloor new_position);
void update_floor_movement_display(void);

uint8_t get_traveller_colour(ElevatorFloor dest_floor);
void check_traveller_pickup(void);
void check_traveller_dropoff(void);
void hardware_pwm_ssd_displays(void);
void clear_all_segments(void);
void display_floor_number(uint8_t floor_num);
void display_direction(int8_t direction);
void update_ssd_values(int8_t direction, ElevatorFloor floor);
uint8_t is_between_floors(ElevatorFloor position);

void play_traveller_placed_sound(void);
void play_pickup_dropoff_sound(void);
void stop_sound(void);
void update_sound(void);
void setup_timer2_for_sound(void);
void start_timer_sound(uint16_t frequency);
void stop_timer_sound(void);
void start_door_animation(void);
void update_door_animation(void);
void stop_door_animation(void);
void set_door_leds(uint8_t l0, uint8_t l1, uint8_t l2, uint8_t l3);

void initialize_traveller_queue(void);
int8_t add_traveller_to_queue(ElevatorFloor start_floor, ElevatorFloor dest_floor);
void remove_traveller_from_queue(uint8_t index);
int8_t find_next_traveller_to_pickup(void);
ElevatorFloor find_next_destination(void);

void init_adc(void);
uint16_t read_adc(uint8_t channel);
void check_service_mode_switch(void);
int8_t get_joystick_direction_and_speed(uint16_t *speed_delay);
void handle_service_mode(void);

/* Main */

int main(void) {
    // Setup hardware and call backs. This will turn on 
    // interrupts.
    initialise_hardware();
    
    // Show the splash screen message. Returns when display is complete
    start_screen();
    
    // Start elevator controller software
    start_elevator_emulator();
}

/* Internal Function Definitions */

void initialise_hardware(void) {
    ledmatrix_setup();
    init_button_interrupts();
    init_serial_stdio(19200, 0);
    init_timer0();

    DDRA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D);
    DDRD |= (1 << SSD_SEG_E) | (1 << SSD_SEG_F) | (1 << SSD_SEG_G) | (1 << CC_PIN) | (1 << SSD_DP);
    
    DDRC &= ~((1 << PC0) | (1 << PC1) | (1 << PC2));
    PORTC |= (1 << PC0) | (1 << PC1) | (1 << PC2);
    
    DDRC |= (1 << PC3);
    PORTC &= ~(1 << PC3);
    
    DDRC |= (1 << LED_L0) | (1 << LED_L1) | (1 << LED_L2) | (1 << LED_L3);
    PORTC &= ~((1 << LED_L0) | (1 << LED_L1) | (1 << LED_L2) | (1 << LED_L3));
    
    setup_timer2_for_sound();

    init_adc();
    
    DDRA &= ~(1 << SERVICE_MODE_SW);
    PORTA |= (1 << SERVICE_MODE_SW);

    sei();
}

void start_screen(void) {
    // Clear terminal screen and output student information
    clear_terminal();
    
    // Move cursor and display student number
    move_terminal_cursor(10, 10);
    printf_P(PSTR("Student Number: 48042244"));
    
    // Move cursor and display student name  
    move_terminal_cursor(10, 12);
    printf_P(PSTR("Name: Mitchell Wishart"));
    
    // Add instructions for user
    move_terminal_cursor(10, 14);
    printf_P(PSTR("Press 's' or any button to start elevator emulator"));
    
    // Show start screen on LED matrix
    start_display();
    
    // Animation variables for LED matrix
    uint32_t doors_frame_time = 0;
    uint32_t interval_delay = 150;
    uint8_t frame = 0;
    uint8_t doors_opening_closing = 1;
    
    while(1) {
        if (get_current_time() - doors_frame_time > interval_delay) {
            start_display_animation(frame);
            doors_frame_time = get_current_time();
            if (doors_opening_closing) {
                interval_delay = 150;
                frame++;
                if (frame == 1) interval_delay = 2000;
                if (frame == 3) doors_opening_closing = 0;
            } else {
                interval_delay = 150;
                frame--;
                if (frame == 2) interval_delay = 500;
                if (frame == 0) doors_opening_closing = 1;
            }
        }
        char serial_input = -1;
        if (serial_input_available()) {
            serial_input = fgetc(stdin);
        }
        if (serial_input == 's' || serial_input == 'S') {
            break;
        }
        int8_t btn = button_pushed();
        if (btn != NO_BUTTON_PUSHED) {
            break;
        }
    }
    clear_terminal();
}

void setup_timer2_for_sound(void) {
    TCCR2A = (1 << WGM21);  
    TCCR2B = 0;            
    TIMSK2 |= (1 << OCIE2A);
}

ISR(TIMER2_COMPA_vect) {
    if (sound_playing > 0) {
        if (sound_toggle) {
            PORTC |= (1 << PC3); 
        } else {
            PORTC &= ~(1 << PC3);
        }
        sound_toggle = !sound_toggle;
    }
}

void start_timer_sound(uint16_t frequency) {
    
    if (frequency == 3000) {
        OCR2A = 21;
        TCCR2B = (1 << CS22);
    } else if (frequency == 500) {
        OCR2A = 31;
        TCCR2B = (1 << CS22) | (1 << CS21);
    }
    
    sound_toggle = 0;
}

void init_adc(void) {
    ADMUX = (1 << REFS0);
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
}


uint16_t read_adc(uint8_t channel) {
    ADMUX = (ADMUX & 0xF0) | (channel & 0x0F);
   
    ADCSRA |= (1 << ADSC);
    
    while (ADCSRA & (1 << ADSC));
    
    return ADC;
}

void check_service_mode_switch(void) {
    static uint8_t last_service_mode = 0;
    uint8_t switch_on = !(PINA & (1 << SERVICE_MODE_SW));
    
    uint8_t desired_mode = switch_on ? 1 : 0;
    
    if (desired_mode != service_mode_active) {
        if (current_traveller_index == -1) {
            service_mode_active = desired_mode;
            
            move_terminal_cursor(1, 8);
            if (service_mode_active) {
                printf("SERVICE MODE: Manual joystick control active        ");
            } else {
                printf("AUTOMATIC MODE: Normal elevator operation restored   ");
                
                if (total_travellers > 0) {
                    destination = find_next_destination();
                    if (destination != UNDEF_FLOOR && destination != current_position) {
                        move_terminal_cursor(1, 9);
                        printf("RESUMING: Going to pick up traveller on floor %d    ", destination/4);
                    }
                }
            }
        } else {
            if (desired_mode == 1 && last_service_mode == 0) {
                move_terminal_cursor(1, 8);
                printf("WAITING: Will enter service mode after dropoff      ");
            } else if (desired_mode == 0 && last_service_mode == 1) {
                move_terminal_cursor(1, 8);
                printf("WAITING: Will enter automatic mode after dropoff    ");
            }
        }
    }
    
    last_service_mode = service_mode_active;
}

int8_t get_joystick_direction_and_speed(uint16_t *speed_delay) {
    joystick_value = read_adc(JOYSTICK_Y_PIN);
    
    int16_t deviation = joystick_value - JOYSTICK_CENTER;
    
    if (abs(deviation) < JOYSTICK_DEADZONE) {
        *speed_delay = 0;
        return 0;
    }
    
    if (deviation > 0) {
        if (abs(deviation) > JOYSTICK_FAST_THRESHOLD) {
            *speed_delay = 150;
        } else if (abs(deviation) > JOYSTICK_SLOW_THRESHOLD) {
            *speed_delay = 250;
        } else {
            *speed_delay = 0;
            return 0;
        }
        return 1;
    } else {
        if (abs(deviation) > JOYSTICK_FAST_THRESHOLD) {
            *speed_delay = 150;
        } else if (abs(deviation) > JOYSTICK_SLOW_THRESHOLD) {
            *speed_delay = 250;
        } else {
            *speed_delay = 0;
            return 0;
        }
        return -1;
    }
}

void handle_service_mode(void) {
    static uint32_t last_move_time = 0;
    uint16_t move_delay;
    int8_t direction = get_joystick_direction_and_speed(&move_delay);
    
    if (direction != 0 && move_delay > 0 && 
        (get_current_time() - last_move_time > move_delay)) {
        
        ElevatorFloor old_position = current_position;
        ElevatorFloor new_position = current_position;

        if (direction > 0) {  // Up
            new_position = current_position + 1;
        } else { 
            new_position = current_position - 1;
        }
        
        if (new_position < FLOOR_0) {
            new_position = FLOOR_0;
        } else if (new_position > FLOOR_3) {
            new_position = FLOOR_3;
        }
        
        if (new_position != current_position) {
            current_position = new_position;
            
            int8_t display_direction = (new_position > old_position) ? 1 : -1;
            update_ssd_values(display_direction, current_position);
            
            move_terminal_cursor(1, 1);
            printf("SERVICE MODE - Current floor: ");
            if (current_position % 4 == 0) {
                printf("%d", current_position / 4);
            } else {
                printf("%d.%d", current_position / 4, (current_position % 4));
            }
            printf("   Joystick: %d   Direction: %s          ", 
                   joystick_value, 
                   (direction > 0) ? "Up" : (direction < 0) ? "Down" : "Stop");
            
            last_move_time = get_current_time();
        }
    } else if (direction == 0) {
        update_ssd_values(0, current_position);
    }
}


void initialize_traveller_queue(void) {
    for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
        travellers[i].start_floor = UNDEF_FLOOR;
        travellers[i].dest_floor = UNDEF_FLOOR;
        travellers[i].active = false;
        travellers[i].queue_position = 0;
    }
    current_traveller_index = -1;
    total_travellers = 0;
    target_traveller_index = -1;
}

int8_t add_traveller_to_queue(ElevatorFloor start_floor, ElevatorFloor dest_floor) {
    for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
        if (!travellers[i].active) {
            travellers[i].start_floor = start_floor;
            travellers[i].dest_floor = dest_floor;
            travellers[i].active = true;
            
            uint8_t queue_pos = 0;
            for (uint8_t j = 0; j < MAX_TRAVELLERS; j++) {
                if (j != i &&
                    travellers[j].active && 
                    travellers[j].start_floor == start_floor) {
                    queue_pos++;
                }
            }
            travellers[i].queue_position = queue_pos;
            total_travellers++;
            
            return i;
        }
    }
    return -1;
}

void remove_traveller_from_queue(uint8_t index) {
    if (index < MAX_TRAVELLERS && travellers[index].active) {
        ElevatorFloor removed_floor = travellers[index].start_floor;
        uint8_t removed_queue_pos = travellers[index].queue_position;
        
        travellers[index].active = false;
        travellers[index].start_floor = UNDEF_FLOOR;
        travellers[index].dest_floor = UNDEF_FLOOR;
        total_travellers--;
        
        for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
            if (travellers[i].active && 
                travellers[i].start_floor == removed_floor && 
                travellers[i].queue_position > removed_queue_pos) {
                travellers[i].queue_position--;
            }
        }
    }
}

int8_t find_next_traveller_to_pickup(void) {
    for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
        if (travellers[i].active && 
            travellers[i].start_floor == current_position && 
            i != current_traveller_index) {
            return i;
        }
    }
    
    return -1;
}

ElevatorFloor find_next_destination(void) {
    if (current_traveller_index >= 0) {
        return travellers[current_traveller_index].dest_floor;
    }
    
    for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
        if (travellers[i].active && travellers[i].start_floor != UNDEF_FLOOR) {
            target_traveller_index = i;
            return travellers[i].start_floor;
        }
    }
    
    target_traveller_index = -1;
    return UNDEF_FLOOR;
}

void set_door_leds(uint8_t l0, uint8_t l1, uint8_t l2, uint8_t l3) {
    if (l0) PORTC |= (1 << LED_L0); else PORTC &= ~(1 << LED_L0);
    if (l1) PORTC |= (1 << LED_L1); else PORTC &= ~(1 << LED_L1);
    if (l2) PORTC |= (1 << LED_L2); else PORTC &= ~(1 << LED_L2);
    if (l3) PORTC |= (1 << LED_L3); else PORTC &= ~(1 << LED_L3);
}

void start_door_animation(void) {
    if (door_animation_active == 0) {
        door_animation_active = 1;
        door_animation_start_time = get_current_time();
        door_animation_phase = 0;
        
        set_door_leds(0, 1, 1, 0);
    }
}

void update_door_animation(void) {
    if (door_animation_active) {
        uint32_t elapsed = get_current_time() - door_animation_start_time;
        
        if (elapsed < 200) {
            if (door_animation_phase != 0) {
                door_animation_phase = 0;
                set_door_leds(0, 1, 1, 0);
            }
        } else if (elapsed < 400) {
            if (door_animation_phase != 1) {
                door_animation_phase = 1;
                set_door_leds(1, 0, 0, 1);
            }
        } else if (elapsed < 600) {
            if (door_animation_phase != 2) {
                door_animation_phase = 2;
                set_door_leds(0, 1, 1, 0);
            }
        } else {
            stop_door_animation();
        }
    }
}

void stop_door_animation(void) {
    door_animation_active = 0;
    door_animation_phase = 0;
    set_door_leds(0, 0, 0, 0);
}

void stop_timer_sound(void) {
    TCCR2B = 0;
    PORTC &= ~(1 << PC3);
    sound_toggle = 0;
}

void play_traveller_placed_sound(void) {
    if (sound_playing == 0) {
        sound_playing = 1;
        sound_duration = 50;
        sound_start_time = get_current_time();
        start_timer_sound(3000);
    }
}

void play_pickup_dropoff_sound(void) {
    if (sound_playing == 0) {
        sound_playing = 2;
        sound_duration = 100;
        sound_start_time = get_current_time();
        start_timer_sound(500);
    }
}

void stop_sound(void) {
    stop_timer_sound();
    sound_playing = 0;
}

void update_sound(void) {
    if (sound_playing > 0) {
        if (get_current_time() - sound_start_time >= sound_duration) {
            stop_sound();
        }
    }
}

uint8_t is_between_floors(ElevatorFloor position) {
    return (position % 4 != 0);
}

void clear_all_segments(void) {
    PORTA &= ~((1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D));
    PORTD &= ~((1 << SSD_SEG_E) | (1 << SSD_SEG_F) | (1 << SSD_SEG_G) | (1 << SSD_DP));
}

void display_floor_number(uint8_t floor_num) {
    switch (floor_num) {
        case 0:
            PORTA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D);
            PORTD |= (1 << SSD_SEG_E) | (1 << SSD_SEG_F);
            break;
        case 1:
            PORTA |= (1 << SSD_SEG_B) | (1 << SSD_SEG_C);
            break;
        case 2:
            PORTA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_D);
            PORTD |= (1 << SSD_SEG_E) | (1 << SSD_SEG_G);
            break;
        case 3:
            PORTA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D);
            PORTD |= (1 << SSD_SEG_G);
            break;
    }
    
    if (is_between_floors(stored_floor)) {
        PORTD |= (1 << SSD_DP);
    }
}

void display_direction(int8_t direction) {
    if (direction > 0) {
        PORTA |= (1 << SSD_SEG_A);
    } else if (direction < 0) {  
        PORTA |= (1 << SSD_SEG_D);
    } else {
        PORTD |= (1 << SSD_SEG_G);
    }
}

void hardware_pwm_ssd_displays(void) {
    static uint32_t last_switch_time = 0;
    static uint8_t current_display = 0;

    if (get_current_time() - last_switch_time > 1) {
        
        if (current_display == 0) {
            PORTD |= (1 << CC_PIN);
            
            PORTA &= ~((1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D));
            PORTD &= ~((1 << SSD_SEG_E) | (1 << SSD_SEG_F) | (1 << SSD_SEG_G) | (1 << SSD_DP));
            
            if (stored_direction > 0) {
                PORTA |= (1 << SSD_SEG_A);
            } else if (stored_direction < 0) {
                PORTA |= (1 << SSD_SEG_D);
            } else {
                PORTD |= (1 << SSD_SEG_G);
            }
            
        } else {
            PORTD &= ~(1 << CC_PIN);
            
            PORTA &= ~((1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D));
            PORTD &= ~((1 << SSD_SEG_E) | (1 << SSD_SEG_F) | (1 << SSD_SEG_G) | (1 << SSD_DP));
            
            uint8_t floor_num;
            
            if (stored_floor % 4 == 0) {
                floor_num = stored_floor / 4;
            } else {
                floor_num = stored_floor / 4;
            }
            switch (floor_num) {
                case 0:
                    PORTA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D);
                    PORTD |= (1 << SSD_SEG_E) | (1 << SSD_SEG_F);
                    break;
                case 1:
                    PORTA |= (1 << SSD_SEG_B) | (1 << SSD_SEG_C);
                    break;
                case 2:
                    PORTA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_D);
                    PORTD |= (1 << SSD_SEG_E) | (1 << SSD_SEG_G);
                    break;
                case 3:
                    PORTA |= (1 << SSD_SEG_A) | (1 << SSD_SEG_B) | (1 << SSD_SEG_C) | (1 << SSD_SEG_D);
                    PORTD |= (1 << SSD_SEG_G);
                    break;
            }
            
            if (is_between_floors(stored_floor)) {
                PORTD |= (1 << SSD_DP);
            }
        }
        
        current_display = 1 - current_display;
        last_switch_time = get_current_time();
    }
}

void update_ssd_values(int8_t direction, ElevatorFloor floor) {
    stored_direction = direction;
    stored_floor = floor;
}

ElevatorFloor read_destination_from_switches(void) {
    uint8_t switch_value = 0;
    if (!(PINC & (1 << PC0))) switch_value |= 1;
    if (!(PINC & (1 << PC1))) switch_value |= 2;

    switch (switch_value) {
        case 0: return FLOOR_0;
        case 1: return FLOOR_1;
        case 2: return FLOOR_2;
        case 3: return FLOOR_3;
        default: return FLOOR_0;
    }
}

uint8_t get_traveller_colour(ElevatorFloor dest_floor) {
    if (dest_floor == FLOOR_0) {
        return TRAVELLER_TO_0;
    }
    if (dest_floor == FLOOR_1) {
        return TRAVELLER_TO_1;
    }
    if (dest_floor == FLOOR_2) {
        return TRAVELLER_TO_2;
    }
    if (dest_floor == FLOOR_3) {
        return TRAVELLER_TO_3;
    }
    
    return TRAVELLER_TO_0;
}

void check_floor_movement(ElevatorFloor old_position, ElevatorFloor new_position) {
    uint8_t old_floor = old_position / 4;
    uint8_t new_floor = new_position / 4;

    if (old_floor != new_floor) {
        
        if (current_traveller_index >= 0) {
            floors_with_traveller++;
        } else {
            floors_without_traveller++;
        }
        
        update_floor_movement_display();
    }
}

void update_floor_movement_display(void) {
    move_terminal_cursor(1, 2);
    printf("Floors with traveller: %d   Floors without traveller: %d    ", 
           floors_with_traveller, floors_without_traveller);
}

void check_traveller_pickup(void) {
    if (current_traveller_index == -1) {
        if (target_traveller_index >= 0 && 
            travellers[target_traveller_index].active &&
            travellers[target_traveller_index].start_floor == current_position) {
            
            current_traveller_index = target_traveller_index;
            destination = travellers[target_traveller_index].dest_floor;
            target_traveller_index = -1;
            
            ElevatorFloor pickup_floor = travellers[current_traveller_index].start_floor;
            uint8_t pickup_queue_pos = travellers[current_traveller_index].queue_position;
            
            for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
                if (travellers[i].active && 
                    travellers[i].start_floor == pickup_floor && 
                    travellers[i].queue_position > pickup_queue_pos &&
                    i != current_traveller_index) {
                    travellers[i].queue_position--;
                }
            }
            
            play_pickup_dropoff_sound();
            start_door_animation();
            move_terminal_cursor(1, 3);
            printf("Pickup: Floor %d->%d (Raw: %d->%d)                    ", 
                   current_position/4, travellers[current_traveller_index].dest_floor/4,
                   current_position, travellers[current_traveller_index].dest_floor);
        }
    }
}

void check_traveller_dropoff(void) {
    if (current_traveller_index >= 0 && 
        current_position == travellers[current_traveller_index].dest_floor) {
        
        travellers[current_traveller_index].active = false;
        travellers[current_traveller_index].start_floor = UNDEF_FLOOR;
        travellers[current_traveller_index].dest_floor = UNDEF_FLOOR;
        total_travellers--;
        
        current_traveller_index = -1;
        
        destination = find_next_destination();
        if (destination == UNDEF_FLOOR) {
            destination = current_position;
        }
        
        play_pickup_dropoff_sound();
        start_door_animation();
        
        move_terminal_cursor(1, 3);
        printf("Dropoff at floor %d! Next: %d (Raw: %d->%d)     ", 
               current_position/4, destination/4, current_position, destination);
    }
}

void start_elevator_emulator(void) {
    uint16_t slow_delay = 250;
    uint16_t fast_delay = 150;
    uint16_t move_delay;
    
    static ElevatorFloor last_reported_position = UNDEF_FLOOR;
    static int8_t last_direction = 0;

    clear_terminal();
    update_floor_movement_display();
    initialise_display();
    initialize_traveller_queue();
    
    (void)button_pushed();
    clear_serial_input_buffer();

    time_since_move = get_current_time();
    
    current_position = FLOOR_0;
    destination = FLOOR_0;

    update_ssd_values(last_direction, current_position);
    
    while(true) {
        check_service_mode_switch();
        
        if (service_mode_active) {
            handle_service_mode();
            
            hardware_pwm_ssd_displays();
            update_sound();
            update_door_animation();
            
            draw_floors();
            draw_elevator();
            
            handle_inputs();
            
        } else {
            if (PINC & (1 << 2)) {
                move_delay = slow_delay;
            } else {
                move_delay = fast_delay;
            }           

            if (get_current_time() - time_since_move > move_delay) {    
                check_traveller_pickup();
                check_traveller_dropoff();
                
                if (!door_animation_active) {
                    
                    if (current_position != last_reported_position || 
                        (destination > current_position ? 1 : (destination < current_position ? -1 : 0)) != last_direction) {
            
                        move_terminal_cursor(1, 1);
                        printf("Current floor: ");
            
                        if (current_position % 4 == 0) {
                            switch (current_position) {
                                case FLOOR_0: printf("0"); break;
                                case FLOOR_1: printf("1"); break;
                                case FLOOR_2: printf("2"); break;
                                case FLOOR_3: printf("3"); break;
                                default: printf("?"); break;
                            }
                        } else {
                            uint8_t display_floor = current_position / 4;
                            printf("%d", display_floor);
                        }
            
                        if (destination > current_position) {
                            printf("   Direction: Up                ");
                            last_direction = 1;
                        } else if (destination < current_position) {
                            printf("   Direction: Down              ");
                            last_direction = -1;
                        } else {
                            printf("   Direction: Stationary        ");
                            last_direction = 0;
                        }
            
                        update_ssd_values(last_direction, current_position);
                        last_reported_position = current_position;
                    }
                        
                    if (destination != current_position) {
                        ElevatorFloor old_position = current_position;
            
                        if (destination - current_position > 0) {
                            current_position++;
                        } else if (destination - current_position < 0) {
                            current_position--;
                        }
            
                        check_floor_movement(old_position, current_position);
                        
                        update_ssd_values(last_direction, current_position);
                    }
                    
                    draw_floors();
                    draw_elevator();  
                    
                    time_since_move = get_current_time();
                }
                else {
                    time_since_move = get_current_time();
                }
            }
            
            hardware_pwm_ssd_displays();
            update_sound();
            update_door_animation();
            
            handle_inputs();
        }
    }
}

void draw_floors(void) {
    static uint8_t last_traveller_state[MAX_TRAVELLERS];
    static uint8_t first_run = 1;
    uint8_t state_changed = 0;
    
    for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
        uint8_t current_state = travellers[i].active ? (travellers[i].queue_position + 1) : 0;
        if (first_run || last_traveller_state[i] != current_state || i == current_traveller_index) {
            state_changed = 1;
            last_traveller_state[i] = current_state;
        }
    }

    if (state_changed || first_run) {
        for (uint8_t x = 4; x < WIDTH; x++) {
            update_square_colour(x, FLOOR_0 + 1, EMPTY_SQUARE);
            update_square_colour(x, FLOOR_1 + 1, EMPTY_SQUARE);
            update_square_colour(x, FLOOR_2 + 1, EMPTY_SQUARE);
            update_square_colour(x, FLOOR_3 + 1, EMPTY_SQUARE);
        }
        
        for (uint8_t i = 0; i < MAX_TRAVELLERS; i++) {
            if (travellers[i].active && 
                travellers[i].start_floor != UNDEF_FLOOR &&
                i != current_traveller_index) {
                
                uint8_t traveller_y = travellers[i].start_floor + 1;
                uint8_t traveller_x = 4 + travellers[i].queue_position;
                
                if (traveller_x < WIDTH) {
                    uint8_t traveller_colour = get_traveller_colour(travellers[i].dest_floor);
                    update_square_colour(traveller_x, traveller_y, traveller_colour);
                }
            }
        }
        first_run = 0;
    }
    
    for (uint8_t i = 0; i < WIDTH; i++) {
        update_square_colour(i, FLOOR_0, FLOOR);
        update_square_colour(i, FLOOR_1, FLOOR);
        update_square_colour(i, FLOOR_2, FLOOR);
        update_square_colour(i, FLOOR_3, FLOOR);
    }
}

void draw_elevator(void) {
    static uint8_t old_position;
    
    int8_t y = 0;
    
    if (old_position > current_position) {
        y = old_position + 3;
    } else if (old_position < current_position) {
        y = old_position + 1;
    }
    if (y % 4 != 0) {
        update_square_colour(1, y, EMPTY_SQUARE);
        update_square_colour(2, y, EMPTY_SQUARE);
    }
    old_position = current_position;
    
    for (uint8_t i = 1; i <= 3; i++) {
        y = current_position + i;
        if (y % 4 != 0) {
            update_square_colour(1, y, ELEVATOR);
            update_square_colour(2, y, ELEVATOR);
        }
    }
}

void handle_inputs(void) {
    uint8_t btn = button_pushed();
    
    if (serial_input_available()) {
        char key = fgetc(stdin);
        
        if (total_travellers < MAX_TRAVELLERS) {
            ElevatorFloor start_floor = UNDEF_FLOOR;
            
            switch (key) {
                case '0': start_floor = FLOOR_0; break;
                case '1': start_floor = FLOOR_1; break;
                case '2': start_floor = FLOOR_2; break;
                case '3': start_floor = FLOOR_3; break;
                default: return;
            }
            
            ElevatorFloor switch_destination = read_destination_from_switches();
            
            if (start_floor == switch_destination) {
                move_terminal_cursor(1, 5);
                printf("Invalid: Start floor cannot equal destination floor!    ");
                return;
            }
            
            int8_t new_index = add_traveller_to_queue(start_floor, switch_destination);
            if (new_index >= 0) {
                play_traveller_placed_sound();
                
                if (destination == current_position) {
                    destination = find_next_destination();
                }
                
                move_terminal_cursor(1, 4);
                printf("Traveller added to queue: Floor %d -> Floor %d (Total: %d)    ", 
                       start_floor/4, switch_destination/4, total_travellers);
                move_terminal_cursor(1, 5);
                printf("                                                           ");
            } else {
                move_terminal_cursor(1, 5);
                printf("Queue is full! Cannot add more travellers.               ");
            }
        } else {
            move_terminal_cursor(1, 5);
            printf("Queue is full! Cannot add more travellers.               ");
        }
    }
    
    if (btn != NO_BUTTON_PUSHED && total_travellers < MAX_TRAVELLERS) {
        ElevatorFloor start_floor = UNDEF_FLOOR;
        
        if (btn == BUTTON0_PUSHED) start_floor = FLOOR_0;
        else if (btn == BUTTON1_PUSHED) start_floor = FLOOR_1;
        else if (btn == BUTTON2_PUSHED) start_floor = FLOOR_2;
        else if (btn == BUTTON3_PUSHED) start_floor = FLOOR_3;
        else return;
        
        ElevatorFloor switch_destination = read_destination_from_switches();
        
        if (start_floor == switch_destination) {
            move_terminal_cursor(1, 5);
            printf("Invalid: Start floor cannot equal destination floor!    ");
            return;
        }
        
        int8_t new_index = add_traveller_to_queue(start_floor, switch_destination);
        if (new_index >= 0) {
            play_traveller_placed_sound();
            
            if (destination == current_position) {
                destination = find_next_destination();
            }
            
            move_terminal_cursor(1, 4);
            printf("Traveller added to queue: Floor %d -> Floor %d (Total: %d)    ", 
                   start_floor/4, switch_destination/4, total_travellers);
            move_terminal_cursor(1, 5);
            printf("                                                           ");
        }
    }
}
